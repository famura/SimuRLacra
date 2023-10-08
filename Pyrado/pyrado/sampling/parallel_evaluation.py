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

import functools
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, remove_all_dr_wrappers
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import (
    NO_SEED,
    _ps_init,
    _ps_run_one_domain_param,
    _ps_run_one_init_state,
    _ps_run_one_reset_kwargs_segment,
)
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence, check_act_equal
from pyrado.spaces.singular import SingularStateSpace


def eval_domain_params(
    pool: SamplerPool,
    env: SimEnv,
    policy: Policy,
    params: List[Dict],
    init_state: Optional[np.ndarray] = None,
    seed: int = NO_SEED,
) -> List[StepSequence]:
    """
    Evaluate a policy on a multidimensional grid of domain parameters.

    :param pool: parallel sampler
    :param env: environment to evaluate in
    :param policy: policy to evaluate
    :param params: multidimensional grid of domain parameters
    :param init_state: initial state of the environment which will be fixed if not set to `None`
    :return: list of rollouts
    """
    # Strip all domain randomization wrappers from the environment
    env = remove_all_dr_wrappers(env, verbose=True)
    if init_state is not None:
        env.init_space = SingularStateSpace(fixed_state=init_state)

    pool.invoke_all(_ps_init, pickle.dumps(env), pickle.dumps(policy))

    # Run with progress bar
    with tqdm(leave=False, file=sys.stdout, unit="rollouts", desc="Sampling") as pb:
        # we set the sub seed to zero since every run will have its personal sub sub seed
        return pool.run_map(
            functools.partial(_ps_run_one_domain_param, eval=True, seed=seed, sub_seed=0), list(enumerate(params)), pb
        )


def eval_nominal_domain(
    pool: SamplerPool, env: SimEnv, policy: Policy, init_states: List[np.ndarray]
) -> List[StepSequence]:
    """
    Evaluate a policy using the nominal (set in the given environment) domain parameters.

    :param pool: parallel sampler
    :param env: environment to evaluate in
    :param policy: policy to evaluate
    :param init_states: initial states of the environment which will be fixed if not set to `None`
    :return: list of rollouts
    """
    # Strip all domain randomization wrappers from the environment
    env = remove_all_dr_wrappers(env)

    pool.invoke_all(_ps_init, pickle.dumps(env), pickle.dumps(policy))

    # Run with progress bar
    with tqdm(leave=False, file=sys.stdout, unit="rollouts", desc="Sampling") as pb:
        return pool.run_map(functools.partial(_ps_run_one_init_state, eval=True), list(enumerate(init_states)), pb)


def eval_randomized_domain(
    pool: SamplerPool, env: SimEnv, randomizer: DomainRandomizer, policy: Policy, init_states: List[np.ndarray]
) -> List[StepSequence]:
    """
    Evaluate a policy in a randomized domain.

    :param pool: parallel sampler
    :param env: environment to evaluate in
    :param randomizer: randomizer used to sample random domain instances, inherited from `DomainRandomizer`
    :param policy: policy to evaluate
    :param init_states: initial states of the environment which will be fixed if not set to `None`
    :return: list of rollouts
    """
    # Randomize the environments
    env = remove_all_dr_wrappers(env)
    env = DomainRandWrapperLive(env, randomizer)

    pool.invoke_all(_ps_init, pickle.dumps(env), pickle.dumps(policy))

    # Run with progress bar
    with tqdm(leave=False, file=sys.stdout, unit="rollouts", desc="Sampling") as pb:
        return pool.run_map(functools.partial(_ps_run_one_init_state, eval=True), list(enumerate(init_states)), pb)


def eval_domain_params_with_segmentwise_reset(
    pool: SamplerPool,
    env_sim: SimEnv,
    policy: Policy,
    segments_real_all: List[List[StepSequence]],
    domain_params_ml_all: List[List[dict]],
    stop_on_done: bool,
    use_rec: bool,
) -> List[List[StepSequence]]:
    """
    Evaluate a policy for a given set of domain parameters, synchronizing the segments' initial states with the given
    target domain segments

    :param pool: parallel sampler
    :param env_sim: environment to evaluate in
    :param policy: policy to evaluate
    :param segments_real_all: all segments from the target domain rollout
    :param domain_params_ml_all: all domain parameters to evaluate over
    :param stop_on_done: if `True`, the rollouts are stopped as soon as they hit the state or observation space
                             boundaries. This behavior is save, but can lead to short trajectories which are eventually
                             padded with zeroes. Chose `False` to ignore the boundaries (dangerous on the real system).
    :param use_rec: `True` if pre-recorded actions have been used to generate the rollouts
    :return: list of segments of rollouts
    """
    # Sample rollouts with the most likely domain parameter sets associated to that observation
    segments_ml_all = []  # all top max likelihood segments for all target domain rollouts
    for idx_r, (segments_real, domain_params_ml) in tqdm(
        enumerate(zip(segments_real_all, domain_params_ml_all)),
        total=len(segments_real_all),
        desc="Sampling",
        file=sys.stdout,
        leave=False,
    ):
        segments_ml = []  # all top max likelihood segments for one target domain rollout
        cnt_step = 0

        # Iterate over target domain segments
        for segment_real in segments_real:
            # Initialize workers
            pool.invoke_all(_ps_init, pickle.dumps(env_sim), pickle.dumps(policy))

            # Run without progress bar
            segments_dp = pool.run_map(
                functools.partial(
                    _ps_run_one_reset_kwargs_segment,
                    init_state=segment_real.states[0, :],
                    len_segment=segment_real.length,
                    stop_on_done=stop_on_done,
                    use_rec=use_rec,
                    idx_r=idx_r,
                    cnt_step=cnt_step,
                    eval=True,
                ),
                domain_params_ml,
            )
            for sdp in segments_dp:
                assert np.allclose(sdp.states[0, :], segment_real.states[0, :])
                if use_rec:
                    check_act_equal(segment_real, sdp, check_applied=hasattr(sdp, "actions_applied"))

            # Increase step counter for next segment, and append all domain parameter segments
            cnt_step += segment_real.length
            segments_ml.append(segments_dp)

        # Append all segments for the current target domain rollout
        segments_ml_all.append(segments_ml)

    return segments_ml_all
