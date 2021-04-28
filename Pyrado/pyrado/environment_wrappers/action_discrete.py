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

import numpy as np
from init_args_serializer import Serializable

from pyrado.environment_wrappers.base import EnvWrapperAct
from pyrado.environments.base import Env
from pyrado.spaces.base import Space
from pyrado.spaces.discrete import DiscreteSpace


class ActDiscreteWrapper(EnvWrapperAct, Serializable):
    """Environment wrapper that converts a one-dimensional continuous into a discrete action space."""

    def __init__(self, wrapped_env: Env, num_bins: int = 2):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param num_bins: number of actions to split the continuous (box) space into
        """

        Serializable._init(self, locals())

        # Invoke base constructor
        super().__init__(wrapped_env)

        # Store parameter and initialize slot for queue
        self._n_actions = num_bins
        self._actions = np.linspace(
            start=self.wrapped_env.act_space.bound_lo, stop=self.wrapped_env.act_space.bound_up, num=num_bins
        )

    def _process_act(self, act: np.ndarray):
        return self._actions[int(act)]

    def _process_act_space(self, space: Space):
        return DiscreteSpace(list(range(self._n_actions)))
