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

import gym
import gym.spaces
import numpy as np
from gym.spaces.box import Box

import pyrado
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_types import RenderMode


class PysimGymWrapper(gym.Env):
    """
    A wrapper for pysim environments exposing an OpenAI gym env.
    
    Do not instantiate this yourself but rather use `gym.make("SimulacraPySimEnv-v0", env=sim_env)`
    where `sim_env` is your instantiated pysim env.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env: SimPyEnv):
        self._wrapped_env = env

        # translate environment parameters
        self.action_space = self.conv_space_from_simulacra(self._wrapped_env.act_space)
        self.observation_space = self.conv_space_from_simulacra(self._wrapped_env.obs_space)

    def step(self, action):
        return self._wrapped_env.step(action)

    def reset(self):
        return self._wrapped_env.reset()

    def render(self, mode="human"):
        self._wrapped_env.render(mode=RenderMode(text=False, video=True, render=False), render_step=1)

    def close(self):
        self._wrapped_env.close()

    def seed(self, seed=None):
        pyrado.set_seed(seed)

    @staticmethod
    def conv_space_from_simulacra(space: Space) -> gym.spaces.Space:
        """Convert a Simulacra space to a gym space

        :param space: A simulacra space
        :raises NotImplementedError: Raised when no conversion is implemented for the space type
        :return:
        """
        if isinstance(space, BoxSpace):
            bounds = space.bounds
            shape = space.shape
            return Box(low=bounds[0].astype(np.float32), high=bounds[1].astype(np.float32), shape=shape)
        raise NotImplementedError
