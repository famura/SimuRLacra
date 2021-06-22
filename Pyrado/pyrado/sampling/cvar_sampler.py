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

from typing import List, Optional

import numpy as np

import pyrado
from pyrado.environments.base import Env
from pyrado.logger.step import LoggerAware
from pyrado.policies.base import Policy
from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.step_sequence import StepSequence


def select_cvar(rollouts, epsilon: float, gamma: float = 1.0):
    """
    Select a subset of rollouts so that their mean discounted return is the CVaR(eps) of the full rollout set.

    :param rollouts: list of rollouts
    :param epsilon: chosen return quantile
    :param gamma: discount factor to compute the discounted return, default is 1 (no discount)
    :return: list of selected rollouts
    """
    # Select epsilon-quantile of returns
    # Do this first since it is easier here, even though we have to compute the returns again
    # To do so, we first sort the paths by their returns
    rollouts.sort(key=lambda ro: ro.discounted_return(gamma))

    # Compute the quantile on a sorted list is easy
    num_sel_ros = round(len(rollouts) * epsilon)

    if num_sel_ros == 0:
        raise pyrado.ValueErr(given=num_sel_ros, g_constraint=0)

    # Only use the selected rollouts
    return rollouts[0:num_sel_ros]


class CVaRSampler(SamplerBase, LoggerAware):
    """
    Samples rollouts to optimize the CVaR of the discounted return.
    This is done by sampling more rollouts, and then only using the epsilon-qunatile of them.
    """

    def __init__(
        self, wrapped_sampler, epsilon: float, gamma: float = 1.0, *, min_rollouts: int = None, min_steps: int = None
    ):
        """
        Constructor

        :param wrapped_sampler: the inner sampler used to sample the full data set
        :param epsilon: quantile of rollouts that will be kept
        :param gamma: discount factor to compute the discounted return, default is 1 (no discount)
        :param min_rollouts: minimum number of complete rollouts to sample
        :param min_steps: minimum total number of steps to sample
        """
        self._wrapped_sampler = wrapped_sampler
        self.epsilon = epsilon
        self.gamma = gamma

        # Call SamplerBase constructor
        super().__init__(min_rollouts=min_rollouts, min_steps=min_steps)

    def set_min_count(self, min_rollouts=None, min_steps=None):
        # Set inner sampler's parameter values (back) to the user-specified number of rollouts / steps
        super().set_min_count(min_rollouts=min_rollouts, min_steps=min_steps)

        # Increase the number of rollouts / steps that will be sampled since we will discard (1 - eps) quantile
        # This modifies the inner samplers parameter values, that's ok since we don't use them afterwards
        if min_rollouts is not None:
            # Expand rollout count to full set
            min_rollouts = int(min_rollouts / self.epsilon)
        if min_steps is not None:
            # Simply increasing the number of steps as done for the rollouts is not identical, however it is goo enough
            min_steps = int(min_steps / self.epsilon)
        self._wrapped_sampler.set_min_count(min_rollouts=min_rollouts, min_steps=min_steps)

    def reinit(self, env: Optional[Env] = None, policy: Optional[Policy] = None):
        # Delegate to inner sampler
        self._wrapped_sampler.reinit(env=env, policy=policy)

    def sample(self) -> List[StepSequence]:
        # Sample full data set
        fullset = self._wrapped_sampler.sample()

        # Log return-based metrics using the full data set
        rets = np.asarray([ro.undiscounted_return() for ro in fullset])
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value("full avg rollout len", np.mean([ro.length for ro in fullset]))
        self.logger.add_value("full avg return", ret_avg)
        self.logger.add_value("full median return", ret_med)
        self.logger.add_value("full std return", ret_std)

        # Return subset
        return select_cvar(fullset, self.epsilon, self.gamma)
