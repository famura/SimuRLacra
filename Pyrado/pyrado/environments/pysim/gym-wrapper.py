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
    """A wrapper for pysim environments exposing a gym env.
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
