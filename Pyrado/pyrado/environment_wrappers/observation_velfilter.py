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

from typing import List, Optional, Tuple

import numpy as np
from init_args_serializer.serializable import Serializable
from scipy import signal

import pyrado
from pyrado.environment_wrappers.base import EnvWrapperObs
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.sim_base import SimEnv


class ObsVelFiltWrapper(EnvWrapperObs, Serializable):
    """Environment wrapper which computes the velocities from the satets given a linear filter"""

    def __init__(
        self,
        wrapped_env: SimEnv,
        mask_pos: Optional[List] = None,
        idcs_pos: Optional[List] = None,
        mask_vel: Optional[List] = None,
        idcs_vel: Optional[List] = None,
        num: Optional[Tuple] = (50, 0),
        den: Optional[Tuple] = (1, 50),
    ):
        """
        Constructor

        :param wrapped_env: environment to wrap, can only be used on `SimEnv` since access to the state is needed, and
                            we don't want to assume that all `RealEnv` can reconstruct this state from the observations.
                            It wouldn't make much sense to wrap a `RealEnv` with this wrapper anyway, since the goal it
                            to mimic the behavior of the real environments velocity filter.
        :param mask_pos: state mask array to select the position quantities in the state space, entries with 1 are kept
        :param idcs_pos: state indices to select, ignored if mask is specified. If the state space is labeled, these
                         labels can be used as indices.
        :param mask_vel: observation mask array to select the velocity quantities in the observation space,
                         entries with 1 are kept
        :param idcs_vel: velocity observation indices to select, ignored if mask is specified. If the observation space
                         is labeled, these labels can be used as indices.
        :param num: continuous-time filter numerator
        :param den: continuous-time filter denominator
        """
        if not isinstance(inner_env(wrapped_env), SimEnv):
            raise pyrado.TypeErr(given=inner_env(wrapped_env), expected_type=SimEnv)

        Serializable._init(self, locals())

        # Call EnvWrapperObs's constructor
        super().__init__(wrapped_env)

        # Parse selections for the positions to be filtered
        if mask_pos is not None:
            # Use explicit mask
            self.mask_pos = np.array(mask_pos, dtype=bool)
            if not self.mask_pos.shape == wrapped_env.state_space.shape:
                raise pyrado.ShapeErr(given=mask_pos, expected_match=wrapped_env.state_space)
        else:
            # Parse indices
            if idcs_pos is None:
                raise pyrado.ValueErr(msg="Either mask or indices must be specified!")
            self.mask_pos = wrapped_env.state_space.create_mask(idcs_pos)

        # Parse selections for the velocities to be replaced by the filtered positions
        if mask_vel is not None:
            # Use explicit mask
            self.mask_vel = np.array(mask_vel, dtype=bool)
            if not self.mask_vel.shape == wrapped_env.obs_space.shape:
                raise pyrado.ShapeErr(given=mask_vel, expected_match=wrapped_env.obs_space)
        else:
            # Parse indices
            if idcs_vel is None:
                raise pyrado.ValueErr(msg="Either mask or indices must be specified!")
            self.mask_vel = wrapped_env.obs_space.create_mask(idcs_vel)

        # Creat the filter and map it to continuous space
        derivative_filter = signal.cont2discrete((num, den), dt=wrapped_env.dt)

        # Initialize discrete filter coefficients and state
        self.b = derivative_filter[0].ravel().astype(np.float32)
        self.a = derivative_filter[1].astype(np.float32)
        self.z = np.zeros((max(len(self.a), len(self.b)) - 1, sum(self.mask_pos)), dtype=np.float32)

    def init_filter(self, init_state):
        """
        Set the initial state of the velocity filter. This is useful when the initial (position) observation has been
        received and it is non-zero. Otherwise the filter would assume a very high initial velocity.

        :param init_state: initial state to set the filter
        """
        if not isinstance(init_state, np.ndarray):
            raise pyrado.TypeErr(given=init_state, expected_type=np.ndarray)

        # Get the initial condition of the filter
        zi = signal.lfilter_zi(self.b, self.a)  # dim = order of the filter = 1

        # Set the filter state
        self.z = np.outer(zi, init_state[self.mask_pos])

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        init_obs = super().reset(init_state=init_state, domain_param=domain_param)

        # Initialize the filter with the current simulation state
        self.init_filter(self.state)

        # Processed observation again
        return self._process_obs(init_obs)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Do the filtering based on the simulation state
        pos = self.state[None, self.mask_pos]
        vel, self.z = signal.lfilter(self.b, self.a, pos, 0, self.z)

        # Replace the velocity entries into the observations
        obs[self.mask_vel] = vel.flatten()
        return obs
