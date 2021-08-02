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
    Do not instantiate this yourself but rather use `gym.make()` like this::

        sim_env = QQubeSwingUpSim(**env_hparams)
        gym.make("SimulacraPySimEnv-v0", env=sim_env)
    """

    metadata = {"render.modes": ["human"]}
    """Supported render modes (currently only human)
    """

    def __init__(self, env: SimPyEnv):
        """Initialize the environment.
        You are not supposed to call this function directly but use `gym.make("SimulacraPySimEnv-v0", env=sim_env)`
        where `sim_env` is your Pyrado environment.

        :param env: Wrapped Pyrado environment
        """
        self._wrapped_env = env

        # translate environment parameters
        self.action_space = self.conv_space_from_simulacra(self._wrapped_env.act_space)
        self.observation_space = self.conv_space_from_simulacra(self._wrapped_env.obs_space)

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        return self._wrapped_env.step(action)

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function does not reset the environment's random
        number generator(s); random variables in the environment's state are
        sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` yields an environment suitable for
        a new episode, independent of previous episodes.

        :returns observation: the initial observation.
        """
        return self._wrapped_env.reset()

    def render(self, mode="human"):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        :param mode: the mode to render with (currently ignored until other modes besides human are implemented)
        """

        self._wrapped_env.render(mode=RenderMode(text=False, video=True, render=False), render_step=1)

    def close(self):
        """Perform any necessary cleanup for the environment.
        Environments will automatically `close()` themselves when
        garbage collected or when the program exits.
        """
        self._wrapped_env.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        :return: list: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
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
