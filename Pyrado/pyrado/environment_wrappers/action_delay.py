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

import pyrado
from pyrado.environment_wrappers.base import EnvWrapperAct
from pyrado.environments.base import Env


class ActDelayWrapper(EnvWrapperAct, Serializable):
    """ Environment wrapper which delays actions by a fixed number of time steps. """

    def __init__(self, wrapped_env: Env, delay: int = 0):
        """
        Constructor

        :param wrapped_env: environment to wrap around (only makes sense from simulation environments)
        :param delay: integer action delay measured in number of time steps
        """
        Serializable._init(self, locals())

        # Invoke base constructor
        super().__init__(wrapped_env)

        # Store parameter and initialize slot for queue
        self._delay = round(delay)  # round returns int
        self._act_queue = []

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, delay: int):
        # Validate and set
        if not delay >= 0:
            raise pyrado.ValueErr(given=delay, ge_constraint="0")
        self._delay = round(delay)  # round returns int

    def _set_wrapper_domain_param(self, domain_param: dict):
        """
        Store the action delay in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        domain_param["act_delay"] = self._delay

    def _get_wrapper_domain_param(self, domain_param: dict):
        """
        Load the action delay from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        # Cast the delay value to int, since randomizer yields ndarrays or Tensors
        self._delay = int(domain_param.get("act_delay", self._delay))

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Adapt _delay to the new act_delay if provided
        if domain_param is not None:
            self._get_wrapper_domain_param(domain_param)

        # Init action queue with the right amount of 0 actions
        self._act_queue = [np.zeros(self.act_space.shape)] * self._delay

        # Call the reset function of the super class and forwards the arguments
        return super().reset(init_state, domain_param)

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        """
        Return the delayed action.

        :param act: commanded action which will be delayed by _delay time steps
        :return: next action that has been commanded _delay time steps before
        """
        if self._delay != 0:
            # Append current action to queue
            self._act_queue.append(act)

            # Retrieve and remove first element
            act = self._act_queue.pop(0)

        # Return modified action
        return act
