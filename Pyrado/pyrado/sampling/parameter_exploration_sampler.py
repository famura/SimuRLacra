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

import itertools
import multiprocessing as mp
import pickle
import sys
from functools import partial
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np
import torch as to
from init_args_serializer import Serializable
from torch.nn.utils.convert_parameters import vector_to_parameters
from tqdm import tqdm

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import (
    DomainRandWrapper,
    DomainRandWrapperBuffer,
    remove_all_dr_wrappers,
)
from pyrado.environment_wrappers.utils import attr_env_get, inner_env, typed_env
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.properties import cached_property


NO_SEED = object()


class ParameterSample(NamedTuple):
    """Stores policy parameters and associated rollouts."""

    params: to.Tensor
    rollouts: List[StepSequence]

    @property
    def mean_undiscounted_return(self) -> float:
        """Get the mean of the undiscounted returns over all rollouts."""
        return np.mean([r.undiscounted_return() for r in self.rollouts]).item()

    @property
    def num_rollouts(self) -> int:
        """Get the number of rollouts."""
        return len(self.rollouts)


class ParameterSamplingResult(Sequence[ParameterSample]):
    """
    Result of a parameter exploration sampling run.
    On one hand, this is a list of ParameterSamples.
    On the other hand, this allows to query combined tensors of parameters and mean returns.
    """

    def __init__(self, samples: Sequence[ParameterSample]):
        """
        Constructor

        :param samples: list of parameter samples
        """
        self._samples = samples

    def __getitem__(self, idx):
        # Get from samples
        res = self._samples[idx]
        if not isinstance(res, ParameterSample):
            # Was a slice, return a wrapped slice
            return ParameterSamplingResult(res)
        # Single item, return it
        return res

    def __len__(self):
        return len(self._samples)

    @cached_property
    def parameters(self) -> to.Tensor:
        """Get all policy parameters as NxP matrix, where N is the number of samples and P is the policy param dim."""
        return to.stack([s.params for s in self._samples])

    @cached_property
    def mean_returns(self) -> np.ndarray:
        """Get all parameter sample means return as a N-dim vector, where N is the number of samples."""
        return np.array([s.mean_undiscounted_return for s in self._samples])

    @cached_property
    def rollouts(self) -> list:
        """Get all rollouts for all samples, i.e. a list of pop_size items, each a list of nom_rollouts rollouts."""
        return [s.rollouts for s in self._samples]

    @cached_property
    def num_rollouts(self) -> int:
        """Get the total number of rollouts for all samples."""
        return int(np.sum([s.num_rollouts for s in self._samples]))


def _pes_init(G, env, policy):
    """Store pickled (and thus copied) environment and policy."""
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)


def _pes_sample_one(G, param, seed: int, sub_seed: int):
    """Sample one rollout with the current setting."""
    num, (pol_param, dom_param, init_state) = param
    vector_to_parameters(pol_param, G.policy.parameters())

    return rollout(
        G.env,
        G.policy,
        reset_kwargs={
            "init_state": init_state,
            "domain_param": dom_param,
        },
        seed=seed,
        sub_seed=sub_seed,
        sub_sub_seed=num,
    )


class ParameterExplorationSampler(Serializable):
    """Parallel sampler for parameter exploration"""

    def __init__(
        self,
        env: Union[SimEnv, EnvWrapper],
        policy: Policy,
        num_init_states_per_domain: int,
        num_domains: int,
        num_workers: int,
        seed: Optional[int] = None,
    ):
        """
        Constructor

        :param env: environment to sample from
        :param policy: policy used for sampling
        :param num_init_states_per_domain: number of rollouts to cover the variance over initial states
        :param num_domains: number of rollouts due to the variance over domain parameters
        :param num_workers: number of parallel samplers
        :param seed: seed value for the random number generators, pass `None` for no seeding; defaults to the last seed
                     that was set with `pyrado.set_seed`
        """
        if not isinstance(num_init_states_per_domain, int):
            raise pyrado.TypeErr(given=num_init_states_per_domain, expected_type=int)
        if num_init_states_per_domain < 1:
            raise pyrado.ValueErr(given=num_init_states_per_domain, ge_constraint="1")
        if not isinstance(num_domains, int):
            raise pyrado.TypeErr(given=num_domains, expected_type=int)
        if num_domains < 1:
            raise pyrado.ValueErr(given=num_domains, ge_constraint="1")

        Serializable._init(self, locals())

        # Check environment for domain randomization wrappers (stops after finding the outermost)
        self._dr_wrapper = typed_env(env, DomainRandWrapper)
        if self._dr_wrapper is not None:
            assert isinstance(inner_env(env), SimEnv)
            # Remove them all from the env chain since we sample the domain parameter later explicitly
            env = remove_all_dr_wrappers(env)

        self.env, self.policy = env, policy
        self.num_init_states_per_domain = num_init_states_per_domain
        self.num_domains = num_domains

        # Set method to spawn if using cuda
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        # Create parallel pool. We use one thread per environment because it's easier.
        self.pool = SamplerPool(num_workers)

        if seed is NO_SEED:
            seed = pyrado.get_base_seed()
        self._seed = seed
        # Initialize with -1 such that we start with the 0-th sample. Incrementing after sampling may cause issues when
        # the sampling crashes and the sample count is not incremented.
        self._sample_count = -1

        # Distribute environments. We use pickle to make sure a copy is created for n_envs = 1
        self.pool.invoke_all(_pes_init, pickle.dumps(self.env), pickle.dumps(self.policy))

    @property
    def num_rollouts_per_param(self) -> int:
        """Get the number of rollouts per policy parameter set."""
        return self.num_init_states_per_domain * self.num_domains

    def _sample_domain_params(self) -> list:
        """Sample domain parameters from the cached domain randomization wrapper."""
        if self._dr_wrapper is None:
            # There was no randomizer, thus do not set any domain parameters
            return [None] * self.num_domains

        elif isinstance(self._dr_wrapper, DomainRandWrapperBuffer) and self._dr_wrapper.buffer is not None:
            # Use buffered domain parameter sets
            idcs = np.random.randint(0, len(self._dr_wrapper.buffer), size=self.num_domains)
            return [self._dr_wrapper.buffer[i] for i in idcs]

        else:
            # Sample new domain parameters (same as in DomainRandWrapperBuffer.fill_buffer)
            self._dr_wrapper.randomizer.randomize(self.num_domains)
            return self._dr_wrapper.randomizer.get_params(-1, fmt="list", dtype="numpy")

    def _sample_one_init_state(self, domain_param: dict) -> Union[np.ndarray, None]:
        """
        Sample an init state for the given domain parameter set(s).
        For some environments, the initial state space depends on the domain parameters, so we need to set them before
        sampling it. We can just reset `self.env` here safely though, since it's not used for anything else.

        :param domain_param: domain parameters to set
        :return: initial state, `None` if no initial state space is defined
        """
        self.env.reset(domain_param=domain_param)
        ispace = attr_env_get(self.env, "init_space")
        if ispace is not None:
            return ispace.sample_uniform()
        else:
            # No init space, no init state
            return None

    def reinit(self, env: Optional[Env] = None, policy: Optional[Policy] = None):
        """
        Re-initialize the sampler.

        :param env: the environment which the policy operates
        :param policy: the policy used for sampling
        """
        # Update env and policy if passed
        if env is not None:
            self.env = env
        if policy is not None:
            self.policy = policy

        # Always broadcast to workers
        self.pool.invoke_all(_pes_init, pickle.dumps(self.env), pickle.dumps(self.policy))

    def sample(self, param_sets: to.Tensor, init_states: Optional[List[np.ndarray]] = None) -> ParameterSamplingResult:
        """
        Sample rollouts for a given set of parameters.

        .. note::
            This method is **not** thread-safe! See for example the usage of `self._sample_count`.

        :param param_sets: sets of policy parameters
        :param init_states: fixed initial states, pass `None` to randomly sample initial states
        :return: data structure containing the policy parameter sets and the associated rollout data
        """
        if init_states is not None and not isinstance(init_states, list):
            pyrado.TypeErr(given=init_states, expected_type=list)

        self._sample_count += 1

        # Sample domain parameter sets
        domain_params = self._sample_domain_params()
        if not isinstance(domain_params, list):
            raise pyrado.TypeErr(given=domain_params, expected_type=list)

        # Sample the initial states for every domain, but reset before. Hence they are associated to their domain.
        if init_states is None:
            init_states = [
                self._sample_one_init_state(dp) for dp in domain_params for _ in range(self.num_init_states_per_domain)
            ]
        else:
            # This is an edge case, but here we need as many init states as num_domains * num_init_states_per_domain
            if not len(init_states) == len(domain_params * self.num_init_states_per_domain):
                raise pyrado.ShapeErr(given=init_states, expected_match=domain_params * self.num_init_states_per_domain)

        # Repeat the sets for the number of initial states per domain
        domain_params *= self.num_init_states_per_domain

        # Explode parameter list for rollouts per param
        all_params = [(p, *r) for p in param_sets for r in zip(domain_params, init_states)]

        # Sample rollouts in parallel
        with tqdm(leave=False, file=sys.stdout, desc="Sampling", unit="rollouts") as pb:
            all_ros = self.pool.run_map(
                partial(_pes_sample_one, seed=self._seed, sub_seed=self._sample_count), list(enumerate(all_params)), pb
            )

        # Group rollouts by parameters
        ros_iter = iter(all_ros)
        return ParameterSamplingResult(
            [
                ParameterSample(params=p, rollouts=list(itertools.islice(ros_iter, self.num_rollouts_per_param)))
                for p in param_sets
            ]
        )
