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

import pickle
import functools
import numpy as np
import sys
from tqdm import tqdm
from typing import List, Dict, Optional

from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers, DomainRandWrapperLive
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import _ps_init, _ps_run_one_domain_param, _ps_run_one_init_state
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.singular import SingularStateSpace


def eval_domain_params(
    pool: SamplerPool, env: SimEnv, policy: Policy, params: List[Dict], init_state: Optional[np.ndarray] = None
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
        return pool.run_map(functools.partial(_ps_run_one_domain_param, eval=True), params, pb)


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
        return pool.run_map(functools.partial(_ps_run_one_init_state, eval=True), init_states, pb)


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
        return pool.run_map(functools.partial(_ps_run_one_init_state, eval=True), init_states, pb)
