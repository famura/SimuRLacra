# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from copy import deepcopy
from queue import Queue
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch as to
from scipy.spatial.distance import pdist, squareform
from torch import nn
from torch.distributions.kl import kl_divergence
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.initialization import init_param
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler


class SVPG(Algorithm):
    """
    Stein Variational Policy Gradient (SVPG)

    .. seealso::
        [1] Yang Liu, Prajit Ramachandran, Qiang Liu, Jian Peng, "Stein Variational Policy Gradient", arXiv, 2017
    """

    name: str = "svpg"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Env,
        particle: Algorithm,
        max_iter: int,
        num_particles: int,
        temperature: float,
        horizon: int,
        logger: StepLogger = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :type save_dir: pyrado.PathLike
        :param env: the environment which the policy operates
        :type env: Env
        :param particle: the particle to populate with different parameters during training
        :type particle: Algorithm
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :type max_iter: int
        :param num_particles: number of SVPG particles
        :type num_particles: int
        :param temperature: SVPG temperature
        :type temperature: float
        :param horizon: horizon for each particle
        :type horizon: int
        :param logger: defaults to None
        :type logger: StepLogger, optional
        :raises pyrado.TypeErr:
        """

        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy=None, logger=logger)

        # Store the inputs
        self._env = env
        self.num_particles = num_particles
        self.horizon = horizon
        self.temperature = temperature
        self.particle = particle
        self.current_particle = 0

        class OptimizerHook:
            """
            This class mocks the optimizer interface partially to intercept the gradient updates of svpg.
            """

            def __init__(self, particle):
                """Constructor

                :param particle: The blueprint algorithm in which the optimizer is replaced by the mocked one.
                :type particle: Algorithm
                """
                self.optim = particle.optim
                self.buffer = Queue()
                self.particle = particle

            def real_step(self, *args, **kwargs):
                """Call the original optimizer with given args and kwargs."""
                self.optim.step(*args, **kwargs)

            def iter_steps(self) -> Tuple[List, Dict, to.Tensor, to.Tensor]:
                """Generate the steps in the buffer queue.

                :yield: the next step in the queue
                :rtype: Tuple[List, Dict, to.Tensor, to.Tensor]
                """
                while not self.buffer.empty():
                    yield self.buffer.get()

            def empty(self) -> bool:
                """Check if the buffer is empty.

                :return: buffer is empty
                :rtype: bool
                """
                return self.buffer.empty()

            def get_next_step(self) -> Tuple[List, Dict, to.Tensor, to.Tensor]:
                """Retrieve the next step from the queue. It contaisn all calls to the wrapped optimizer which have been intercepted.

                :return: Next buffer step
                :rtype: Tuple[List, Dict, to.Tensor, to.Tensor]
                """
                return self.buffer.get()

            def step(self, *args, **kwargs):
                """Store the args of the mocked call in the queue."""
                self.buffer.put(
                    (
                        args,
                        kwargs,
                        to.clone(self.particle.policy.param_values),
                        to.clone(self.particle.policy.param_grad),
                    )
                )

            def zero_grad(self, *args, **kwargs):
                self.optim.zero_grad(*args, **kwargs)

        self.optims = []
        # Store particle states
        for i in range(self.num_particles):
            self.optims.append(OptimizerHook(self.particle))
        self.particle_states = [particle.__getstate__()] * self.num_particles
        self.particle_policy_states = [particle.policy.param_values] * self.num_particles

        self.particle_steps = [0] * self.num_particles

        for i in range(self.num_particles):
            self.particle.__setstate__(self.particle_states[i])
            self.particle.policy.param_values = self.particle_policy_states[i]
            self.particle.policy.init_param()
            self.particle_states[i] = self.particle.__getstate__()
            self.particle_policy_states[i] = to.clone(self.particle.policy.param_values)
            self.current_particle = i

    @property
    def iter_particles(self):
        """Iterate particles by sequentially loading and yielding them."""
        for i in range(self.num_particles):
            self.particle.__setstate__(self.particle_states[i])
            self.particle.policy.param_values = self.particle_policy_states[i]
            self.particle.optim = self.optims[i]
            self.current_particle = i
            yield self.particle
            self.particle_states[i] = self.particle.__getstate__()
            self.particle_policy_states[i] = to.clone(self.particle.policy.param_values)

    def step(self, snapshot_mode: str, meta_info: dict = None):
        parameters = [[] for i in range(self.num_particles)]
        policy_grads = [[] for i in range(self.num_particles)]
        kwargs = [[] for i in range(self.num_particles)]
        args = [[] for i in range(self.num_particles)]
        for i, particle in enumerate(self.iter_particles):
            particle.step(snapshot_mode="no")
            while not particle.optim.empty():
                args_i, kwargs_i, params, grads = particle.optim.get_next_step()
                policy_grads[i].append(grads)
                parameters[i].append(params)
                args[i].append(args_i)
                kwargs[i].append(kwargs_i)

        # assert all(len(p) == len(parameters[0]) for p in parameters)
        average = True
        if average:
            params = to.stack([to.stack(parameters[i]).mean(axis=0) for i in range(self.num_particles)])
            policy_grds = to.stack([to.stack(policy_grads[i]).mean(axis=0) for i in range(self.num_particles)])
            Kxx, dx_Kxx = self.kernel(params)
            grad_theta = (to.mm(Kxx, policy_grds / self.temperature) + dx_Kxx) / self.num_particles
            for i, particle in enumerate(self.iter_particles):
                particle.policy.param_values = params[i]
                particle.policy.param_grad = grad_theta[i]
                particle.optim.real_step(*args[i][0], **kwargs[i][0])
        else:
            for t_step in tqdm(range(len(parameters[0]))):
                params = to.stack([parameters[idx][t_step] for idx in range(self.num_particles)])
                policy_grds = to.stack([policy_grads[idx][t_step] for idx in range(self.num_particles)])
                Kxx, dx_Kxx = self.kernel(params)
                grad_theta = (to.mm(Kxx, policy_grds / self.temperature) + dx_Kxx) / self.num_particles
                for i, particle in enumerate(self.iter_particles):
                    particle.policy.param_values = parameters[i][t_step]
                    particle.policy.param_grad = grad_theta[i]
                    particle.optim.real_step(*args[i][t_step], **kwargs[i][t_step])

    def kernel(self, X: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        """
        Compute the RBF-kernel and the corresponding derivatives.

        :param X: the tensor to compute the kernel from
        :return: the kernel and its derivatives
        """
        X_np = X.cpu().data.numpy()  # use numpy because torch median is flawed
        pairwise_dists = squareform(pdist(X_np)) ** 2
        assert pairwise_dists.shape[0] == self.num_particles

        # Median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(self.num_particles + 1))

        # Compute RBF Kernel
        kernel = to.exp(-to.from_numpy(pairwise_dists).to(to.get_default_dtype()) / h ** 2 / 2)

        # Compute kernel gradient
        grads = -kernel.matmul(X)
        kernel_sum = kernel.sum(1)
        for i in range(X.shape[1]):
            grads[:, i] = grads[:, i] + X[:, i].matmul(kernel_sum)
        grads /= h ** 2

        return kernel, grads

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._env, "env.pkl", self.save_dir)
            for idx, p in enumerate(self.iter_particles):
                pyrado.save(p, f"particle_{idx}.pt", self.save_dir, use_state_dict=True)
        else:
            # This algorithm instance is a subroutine of another algorithm
            for idx, p in enumerate(self.iter_particles):
                pyrado.save(
                    p,
                    f"particle_{idx}.pt",
                    self.save_dir,
                    prefix=meta_info.get("prefix", ""),
                    suffix=meta_info.get("suffix", ""),
                    use_state_dict=True,
                )

    def load_snapshot(self, parsed_args) -> Tuple[Env, Policy, dict]:
        env, policy, extra = super().load_snapshot(parsed_args)

        # Algorithm specific
        ex_dir = self._save_dir or getattr(parsed_args, "dir", None)
        for idx, p in enumerate(self.iter_particles):
            extra[f"particle{idx}"] = pyrado.load(f"particle_{idx}.pt", ex_dir, obj=self.particles[idx], verbose=True)

        return env, policy, extra

    def load_particle(self, int: idx):
        """
        Load a specific particle's state into `self.particle`.

        :param idx: index of the particle to load
        """
        self.particle.__setstate__(self.particle_states[idx])
        self.particle.policy.param_values = self.particle_policy_states[idx]
        self.current_particle = idx

    def store_particle(self):
        """Safe the current particle's state."""
        self.particle_states[self.current_particle] = self.particle.__getstate__()
        self.particle_policy_states[self.current_particle] = to.clone(self.particle.policy.param_values)
