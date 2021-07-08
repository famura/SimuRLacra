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
from typing import List, Sequence, Tuple, Union

import numpy as np
from pyrado.policies.initialization import init_param
import torch as to
from torch import nn
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from torch.distributions.kl import kl_divergence

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.a2c import A2C
from pyrado.algorithms.step_based.gae import GAE, ValueFunctionSpace
from pyrado.algorithms.utils import compute_action_statistics
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


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
        :param env: the environment which the policy operates
        :param particle_hparam: hyper-parameters for particle template construction
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_particles: number of distinct particles
        :param temperature: the temperature of the SVGD determines how jointly the training takes place
        :param lr: the learning rate for the update of the particles
        :param horizon: horizon for each particle
        :param std_init: initial standard deviation for the exploration
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_workers: number of environments for parallel sampling
        :param serial: serial mode can be switched off which can be used to partly control the flow of SVPG from outside
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
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

        class OptimizerHook:
            def __init__(self, particle):
                self.optim = particle.optim
                self.buffer = Queue()
                self.particle = particle

            def real_step(self, *args, **kwargs):
                self.optim.step(*args, **kwargs)

            def iter_steps(self):
                while not self.buffer.empty():
                    yield self.buffer.get()

            def empty(self):
                return self.buffer.empty()

            def get_next_step(self):
                return self.buffer.get()

            def step(self, *args, **kwargs):
                self.buffer.put((args, kwargs, to.clone(self.particle.policy.param_values), to.clone(self.particle.policy.param_grad)))

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



    @property
    def iter_particles(self):
        for i in range(self.num_particles):
            self.particle.__setstate__(self.particle_states[i])
            self.particle.policy.param_values = self.particle_policy_states[i]
            self.particle.optim = self.optims[i]
            yield self.particle
            self.particle_states[i] = self.particle.__getstate__()
            self.particle_policy_states[i] = to.clone(self.particle.policy.param_values)


    def step(self, snapshot_mode: str, meta_info: dict = None):
        print('Begin step')

        for i, particle in enumerate(self.iter_particles):
            print(particle.policy.param_values[:5])

        parameters = [[]] * self.num_particles
        policy_grads = [[]] * self.num_particles
        args = [[]] * self.num_particles
        kwargs = [[]] * self.num_particles
        for i, particle in enumerate(self.iter_particles):
            particle.step(snapshot_mode="no")
            while not particle.optim.empty():
                args_i, kwargs_i, params, grads = particle.optim.get_next_step()
                print(i, '>>>>>',params[:5])
                policy_grads[i].append(to.tensor(grads.detach()))
                parameters[i].append(to.tensor(params.detach()))
                args[i].append(args_i)
                kwargs[i].append(kwargs_i)

        assert all(len(p) == len(parameters[0]) for p in parameters)

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
        print(X)
        X_np = X.cpu().data.numpy()  # use numpy because torch median is flawed
        pairwise_dists = squareform(pdist(X_np)) ** 2
        print(pairwise_dists)
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
            for idx, p in enumerate(self.particles):
                pyrado.save(p, f"particle_{idx}.pt", self.save_dir, use_state_dict=True)
        else:
            # This algorithm instance is a subroutine of another algorithm
            for idx, p in enumerate(self.particles):
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
        for idx, p in enumerate(self.particles):
            extra[f"particle{idx}"] = pyrado.load(f"particle_{idx}.pt", ex_dir, obj=self.particles[idx], verbose=True)

        return env, policy, extra
