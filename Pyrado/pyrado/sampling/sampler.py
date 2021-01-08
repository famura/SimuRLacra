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

from abc import ABC, abstractmethod
from typing import List

from pyrado.sampling.step_sequence import StepSequence
from pyrado.environments.base import Env
from pyrado.policies.base import Policy


class SamplerBase(ABC):
    """
    A sampler generates a list of rollouts in some unspecified way.

    Since the sampling might occur in parallel, there is no way to reliably generate an exact amount of samples.
    The sampler can however guarantee a minimum amount of samples to be available. The sampler does not discard any
    samples on it's own, all sampled data will be returned.
    There are two ways to regulate the sampling process:
    1. the minimum number of rollouts
    2. the minimum number of steps in all rollouts

    At least one of these bounds must be specified. If both are set, the sampler will only terminate once both are
    fulfilled.
    """

    def __init__(self, *, min_rollouts: int = None, min_steps: int = None):
        """
        Constructor

        :param min_rollouts: minimum number of complete rollouts to sample
        :param min_steps: minimum total number of steps to sample
        """
        self.min_rollouts = None
        self.min_steps = None
        self.set_min_count(min_rollouts, min_steps)

    def set_min_count(self, min_rollouts: int = None, min_steps: int = None):
        """
        Adapt the sampling boundaries.

        :param min_rollouts: minimum number of complete rollouts to sample
        :param min_steps: minimum total number of steps to sample
        """
        assert min_rollouts is not None or min_steps is not None, "At least one limit must be given"
        self.min_rollouts = min_rollouts
        self.min_steps = min_steps

    @abstractmethod
    def reinit(self, env: Env = None, policy: Policy = None):
        """
        Reset the sampler after changes were made to the environment or the policy, optionally replacing one of them.

        Most samplers will be implemented in parallel, so if there are changes to the environment or the policy,
        they will not automatically propagate to all processes. This method exists as a workaround; call it to
        force a reinitialization of environment and policy in all subprocesses.

        Note that you don't need to call this if the policy parameters change, since that is to be expected between
        sampling runs, the sample() method takes care of this on it's own.

        You can use the env and policy parameters to completely replace the stored environment or policy.

        :param env: new environment to use, or None to keep the old one
        :param policy: new policy to use, or None to keep the old one
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> List[StepSequence]:
        """
        Generate a list of rollouts. This method works exactly as specified in the class description.

        :return: sampled rollouts
        """
        raise NotImplementedError
