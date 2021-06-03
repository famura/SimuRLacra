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

import pickle
import sys
from functools import partial
from itertools import product
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
import torch.multiprocessing as mp
from init_args_serializer import Serializable
from tqdm import tqdm

import pyrado
from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.box import InfBoxSpace


def _ps_init(G, env, policy):
    """Store pickled (and thus copied) environment and policy."""
    G.env = pickle.loads(env)
    G.policy = pickle.loads(policy)


def _ps_update_policy(G, state):
    """Update policy state_dict."""
    G.policy.load_state_dict(state)


def _ps_sample_one(G, num: int, eval: bool, seed: int, sub_seed: int):
    """
    Sample one rollout and return step count if counting steps, rollout count (1) otherwise.
    This function is used when a minimum number of steps was given.
    """
    ro = rollout(G.env, G.policy, eval=eval, seed=seed, sub_seed=sub_seed, sub_sub_seed=num)
    return ro, len(ro)


def _ps_run_one(G, num: int, eval: bool, seed: int, sub_seed: int):
    """
    Sample one rollout without specifying the initial state or the domain parameters.
    This function is used when a minimum number of rollouts was given.
    """
    return rollout(G.env, G.policy, eval=eval, seed=seed, sub_seed=sub_seed, sub_sub_seed=num)


def _ps_run_one_init_state(G, init_state: Tuple[int, np.ndarray], eval: bool, seed: int, sub_seed: int):
    """
    Sample one rollout with given init state.
    This function is used when a minimum number of rollouts was given.
    """
    num, init_state = init_state
    return rollout(
        G.env,
        G.policy,
        eval=eval,
        seed=seed,
        sub_seed=sub_seed,
        sub_sub_seed=num,
        reset_kwargs=dict(init_state=init_state),
    )


def _ps_run_one_domain_param(G, domain_param: Tuple[int, dict], eval: bool, seed: int, sub_seed: int):
    """
    Sample one rollout with given domain parameters.
    This function is used when a minimum number of rollouts was given.
    """
    num, domain_param = domain_param
    return rollout(
        G.env,
        G.policy,
        eval=eval,
        seed=seed,
        sub_seed=sub_seed,
        sub_sub_seed=num,
        reset_kwargs=dict(domain_param=domain_param),
    )


def _ps_run_one_reset_kwargs(
    G, reset_kwargs: Tuple[int, Tuple[np.ndarray, dict]], eval: bool, seed: int, sub_seed: int
):
    """
    Sample one rollout with given init state and domain parameters, passed as a tuple for simplicity at the other end.
    This function is used when a minimum number of rollouts was given.
    """
    num, reset_kwargs = reset_kwargs
    return rollout(
        G.env,
        G.policy,
        eval=eval,
        seed=seed,
        sub_seed=sub_seed,
        sub_sub_seed=num,
        reset_kwargs=dict(init_state=reset_kwargs[0], domain_param=reset_kwargs[1]),
    )


def _ps_run_one_reset_kwargs_segment(
    G,
    domain_param: dict,
    init_state: np.ndarray,
    len_segment: int,
    use_rec: bool,
    idx_r: int,
    cnt_step: int,
    eval: bool,
):
    """
    Sample one segments of a rollout with given init state (which originates from a target domain setup) and domain
    parameters, passed as a tuple for simplicity at the other end.
    """
    if not isinstance(domain_param, dict):
        raise pyrado.TypeErr(given=domain_param, expected_type=dict)
    if not isinstance(init_state, np.ndarray):
        raise pyrado.TypeErr(given=init_state, expected_type=np.ndarray)
    if not isinstance(len_segment, int):
        raise pyrado.TypeErr(given=len_segment, expected_type=int)

    # Set the init space of the simulation environment such that we can later set to arbitrary states that could have
    # occurred during the rollout. This is necessary since we are running the evaluation in segments.
    G.env.init_space = InfBoxSpace(shape=G.env.init_space.shape)

    if use_rec:
        # Disabled the policy reset of PlaybackPolicy to do it here manually
        assert G.policy.no_reset
        G.policy.curr_rec = idx_r
        G.policy.curr_step = cnt_step

    return rollout(
        G.env,
        G.policy,
        eval=eval,
        reset_kwargs=dict(init_state=init_state, domain_param=domain_param),
        max_steps=len_segment,
        stop_on_done=False,
    )


class ParallelRolloutSampler(SamplerBase, Serializable):
    """Class for sampling from multiple environments in parallel"""

    def __init__(
        self,
        env,
        policy,
        num_workers: int,
        *,
        min_rollouts: int = None,
        min_steps: int = None,
        show_progress_bar: bool = True,
        seed: int = None,
    ):
        """
        Constructor

        :param env: environment to sample from
        :param policy: policy to act in the environment (can also be an exploration strategy)
        :param num_workers: number of parallel samplers
        :param min_rollouts: minimum number of complete rollouts to sample
        :param min_steps: minimum total number of steps to sample
        :param show_progress_bar: it `True`, display a progress bar using `tqdm`
        :param seed: seed value for the random number generators, pass `None` for no seeding
        """
        Serializable._init(self, locals())
        super().__init__(min_rollouts=min_rollouts, min_steps=min_steps)

        self.env = env
        self.policy = policy
        self.show_progress_bar = show_progress_bar

        # Set method to spawn if using cuda
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        # Create parallel pool. We use one thread per env because it's easier.
        self.pool = SamplerPool(num_workers)

        self._seed = seed
        # Initialize with -1 such that we start with the 0-th sample. Incrementing after sampling may cause issues when
        # the sampling crashes and the sample count is not incremented.
        self._sample_count = -1

        # Distribute environments. We use pickle to make sure a copy is created for n_envs=1
        self.pool.invoke_all(_ps_init, pickle.dumps(self.env), pickle.dumps(self.policy))

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
        self.pool.invoke_all(_ps_init, pickle.dumps(self.env), pickle.dumps(self.policy))

    def sample(
        self,
        init_states: Optional[List[np.ndarray]] = None,
        domain_params: Optional[List[dict]] = None,
        eval: bool = False,
    ) -> List[StepSequence]:
        """
        Do the sampling according to the previously given environment, policy, and number of steps/rollouts.

        .. note::
            This method is **not** thread-safe! See for example the usage of `self._sample_count`.

        :param init_states: initial states forw `run_map()`, pass `None` (default) to sample from the environment's
                            initial state space
        :param domain_params: domain parameters for `run_map()`, pass `None` (default) to not explicitly set them
        :param eval: pass `False` if the rollout is executed during training, else `True`. Forwarded to `rollout()`.
        :return: list of sampled rollouts
        """
        self._sample_count += 1

        # Update policy's state
        self.pool.invoke_all(_ps_update_policy, self.policy.state_dict())

        # Collect samples
        with tqdm(
            leave=False,
            file=sys.stdout,
            desc="Sampling",
            disable=(not self.show_progress_bar),
            unit="steps" if self.min_steps is not None else "rollouts",
        ) as pb:

            if self.min_steps is None:
                if init_states is None and domain_params is None:
                    # Simply run min_rollouts times
                    func = partial(_ps_run_one, eval=eval)
                    arglist = list(range(self.min_rollouts))
                elif init_states is not None and domain_params is None:
                    # Run every initial state so often that we at least get min_rollouts trajectories
                    func = partial(_ps_run_one_init_state, eval=eval)
                    rep_factor = ceil(self.min_rollouts / len(init_states))
                    arglist = list(enumerate(rep_factor * init_states))
                elif init_states is None and domain_params is not None:
                    # Run every domain parameter set so often that we at least get min_rollouts trajectories
                    func = partial(_ps_run_one_domain_param, eval=eval)
                    rep_factor = ceil(self.min_rollouts / len(domain_params))
                    arglist = list(enumerate(rep_factor * domain_params))
                elif init_states is not None and domain_params is not None:
                    # Run every combination of initial state and domain parameter so often that we at least get
                    # min_rollouts trajectories
                    func = partial(_ps_run_one_reset_kwargs, eval=eval)
                    allcombs = list(product(init_states, domain_params))
                    rep_factor = ceil(self.min_rollouts / len(allcombs))
                    arglist = list(enumerate(rep_factor * allcombs))

                # Only minimum number of rollouts given, thus use run_map
                return self.pool.run_map(partial(func, seed=self._seed, sub_seed=self._sample_count), arglist, pb)

            else:
                # Minimum number of steps given, thus use run_collect (automatically handles min_runs=None)
                if init_states is None:
                    return self.pool.run_collect(
                        self.min_steps,
                        partial(_ps_sample_one, eval=eval, seed=self._seed, sub_seed=self._sample_count),
                        collect_progressbar=pb,
                        min_runs=self.min_rollouts,
                    )[0]
                else:
                    raise NotImplementedError
                    # return self.pool.run_collect(
                    #     self.min_steps,
                    #     _ps_run_one_init_state,
                    #     init_states,  # *args
                    #     collect_progressbar=pb,
                    #     min_runs=self.min_rollouts
                    # )[0]
