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
import numpy as np
from collections import deque
from init_args_serializer import Serializable

from pyrado.environment_wrappers.base import EnvWrapper, EnvWrapperAct, EnvWrapperObs
from pyrado.environments.quanser.base import RealEnv


class DownsamplingWrapper(EnvWrapperAct, EnvWrapperObs, Serializable):
    """
    Environment wrapper which downsamples the actions coming from the rollout loop.
    This wrapper is intended to be used with the real Quanser devices, since these are set up to always run on 500Hz,
    i.e. one send-and-receive every 0.002s. When learning in simulation, this requires a lot of samples per rollout,
    which makes learning more time-consuming and difficult (fine tuning the temporal discount factor). In order to
    be able to learn on a lower frequency, e.g. 100Hz, we downsample the actions passed to the real device, i.e. just
    send every factor action.
    .. note:: The observations are not affected! Thus the policy (and the velocity filter) still receive one observation
    per environment step, i.e. send-and-receive when using the real device.
    """

    def __init__(self,
                 wrapped_env: [RealEnv, EnvWrapper],
                 factor: int,
                 obs_filter_fcn: callable = functools.partial(np.mean, axis=0),
                 init_obs: np.ndarray = None):
        """
        Constructor

        :param wrapped_env: environment to wrap around
        :param factor: downsampling factor i.e. number of time steps for which every action should be repeated
        :param obs_filter_fcn: function for processing the observations in the buffer, operates along 0-dimension
        :param init_obs: initial observation to see the buffer, if None the buffer is initialized with zero arrays
        """
        Serializable._init(self, locals())

        # Invoke base constructor
        super().__init__(wrapped_env)

        self._factor = factor
        self._cnt = 0
        self._act_last = None
        if init_obs is None:
            self._obs_buffer = deque([], maxlen=factor)
        else:
            assert isinstance(init_obs, np.ndarray)
            self._obs_buffer = deque([init_obs.copy()], maxlen=factor)
        self._obs_filter_fcn = obs_filter_fcn

    @property
    def factor(self):
        """ Get the downsampling factor. """
        return self._factor

    @factor.setter
    def factor(self, factor: int):
        """ Set the downsampling factor. """
        assert isinstance(factor, int) and factor >= 1
        self._factor = factor
        # Also reset counter
        self._cnt = 0

    def _save_domain_param(self, domain_param: dict):
        """
        Store the downsampling factor in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        # Cast to integer for consistency
        domain_param['downsampling'] = int(self._factor)

    def _load_domain_param(self, domain_param: dict):
        """
        Load the downsampling factor from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        # Cast the factor value to int, since randomizer yields ndarrays or Tensors
        self._factor = int(domain_param.get('downsampling', self._factor))

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Adapt _factor to the new act_downsampling if provided
        if domain_param is not None:
            self._load_domain_param(domain_param)

        # Init last action memory and action counter
        self._act_last = None
        self._cnt = 0

        # Call the reset function of the super class and forwards the arguments
        return super().reset(init_state, domain_param)

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        """
        Return the downsampled action.

        :param act: commanded action
        :return: next action
        """
        if self._factor >= 1:
            if self._cnt%self._factor == 0:
                # Set new action (downsampling does not affect the current action)
                self._act_last = act.copy()
            else:
                # Retrieve last action (downsampling affects the current action)
                act = self._act_last.copy()
            # Increase the counter
            self._cnt += 1

        # Return modified action
        return act

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Process the last factor observations and return a filtered observation.

        :param obs: latest observation from the environment
        :return: filtered observation
        """
        # Add the current observation to the buffer and remove the oldest observation from the buffer
        self._obs_buffer.append(obs)

        # Process the observations in the buffer and return the current estimate (filtered observation)
        return self._obs_filter_fcn(self._obs_buffer)
