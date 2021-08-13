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
    A wrapper for pysim environments exposing a gym env.
    Do not instantiate this yourself but rather use `gym.make()` like this:

    .. code-block:: python

        sim_env = QQubeSwingUpSim(**env_hparams)
        gym.make("SimulacraPySimEnv-v0", env=sim_env)
    """

    metadata = {"render.modes": ["human"]}  # currently only human is supported

    def __init__(self, env: SimPyEnv):
        """
        Initialize the environment. You are not supposed to call this function directly, but use
        `gym.make("SimulacraPySimEnv-v0", env=sim_env)` where `sim_env` is your Pyrado environment.

        :param env: Pyrado environment to wrap
        """
        self._wrapped_env = env

        # Translate environment parameters
        self.action_space = PysimGymWrapper.conv_space_from_pyrado(self._wrapped_env.act_space)
        self.observation_space = PysimGymWrapper.conv_space_from_pyrado(self._wrapped_env.obs_space)

    def step(self, act: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Run one timestep of the environment's dynamics. When the end of episode is reached, you are responsible for
        calling `reset()` to reset this environment's state.

        :param act: action provided by the agent
        :return: observation: agent's current observation of the environment
                 reward: reward returned by the environment
                 done: whether the episode has ended, in which case further `step()` calls will return undefined results
                 info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        return self._wrapped_env.step(act)

    def reset(self) -> np.ndarray:
        """
        Resets the environment to an initial state and returns an initial observation.

        .. note::
            This function does not reset the environment's random number generator(s); random variables in the
            environment's state are sampled independently between multiple calls to `reset()`. In other words, each call
            of `reset()` yields an environment suitable for a new episode, independent of previous episodes.

        :return: the initial observation
        """
        return self._wrapped_env.reset()

    def render(self, mode: str = "human"):
        """
        Renders the environment.
        The set of supported modes varies per environment (some environments do not support rendering).
        By convention, if mode is:
        human: render to the current display or terminal and return nothing. Usually for human consumption.
        rgb_array: return an numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
                   suitable for turning into a video.
        ansi: return a string (str) or StringIO.StringIO containing a terminal-style text representation.
              The text can include newlines and ANSI escape sequences (e.g. for colors).

        :param mode: the mode to render with (currently ignored until other modes besides human are implemented)
        """
        self._wrapped_env.render(mode=RenderMode(text=False, video=True, render=False), render_step=1)

    def close(self):
        """
        Perform any necessary cleanup for the environment. Environments will automatically `close()` themselves when
        garbage collected or when the program exits.
        """
        self._wrapped_env.close()

    def seed(self, seed: Optional[int] = None) -> Optional[List[int]]:
        """
        Sets the seed for this env's random number generator(s).

        :return: list of seeds used in this environments's random number generators. The first value in the list
                 should be the "main" seed, or the value which a reproducer should pass to `seed`. Often, the main
                 seed equals the provided `seed`, but this won't be true if `seed=None`, for example.
        """
        pyrado.set_seed(seed)

    @staticmethod
    def conv_space_from_pyrado(space: Space) -> gym.spaces.Space:
        """
        Convert a Pyrado space to a gym space.

        :param space: Pyrado space to convert
        :return: OpenAI gym space
        """
        if isinstance(space, BoxSpace):
            bounds = space.bounds
            shape = space.shape
            return Box(low=bounds[0].astype(np.float32), high=bounds[1].astype(np.float32), shape=shape)
        raise NotImplementedError
