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

import torch as to
from torch.distributions.uniform import Uniform

from pyrado.policies.base import Policy
from pyrado.policies.recurrent.base import RecurrentPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


class IdlePolicy(Policy):
    """The most simple policy which simply does nothing"""

    name: str = "idle"

    def __init__(self, spec: EnvSpec, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor = None) -> to.Tensor:  # pylint: disable=arguments-differ,unused-argument
        # Observations are ignored
        return to.zeros(self._env_spec.act_space.shape, dtype=to.get_default_dtype(), device=self.device)


class DummyPolicy(Policy):
    """Simple policy which samples random values form the action space"""

    name: str = "dummy"

    def __init__(self, spec: EnvSpec, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor = None) -> to.Tensor:  # pylint: disable=arguments-differ,unused-argument
        # Observations are ignored
        act = to.from_numpy(self.env_spec.act_space.sample_uniform())
        return act.to(dtype=to.get_default_dtype(), device=self.device)


class RecurrentDummyPolicy(RecurrentPolicy):
    """
    Simple recurrent policy which samples random values form the action space and
    always returns hidden states with value zero
    """

    name: str = "rec_cummy"

    def __init__(self, spec: EnvSpec, hidden_size: int, use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param hidden_size: size of the mimic hidden layer
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        low = to.from_numpy(spec.act_space.bound_lo)
        high = to.from_numpy(spec.act_space.bound_up)
        self._distr = Uniform(low, high)
        self._hidden_size = hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor = None, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        # Observations and hidden states are ignored
        return self._distr.sample(), to.zeros(self._hidden_size)

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = "hidden_states") -> to.Tensor:
        raise NotImplementedError()
