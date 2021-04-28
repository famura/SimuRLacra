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

from warnings import warn

import gym.envs
import gym.spaces as gs
import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.data_types import RenderMode


def _to_pyrado_space(gym_space) -> [BoxSpace, DiscreteSpace]:
    """
    Convert a space from OpenAIGym to Pyrado.

    :param gym_space: space object from OpenAIGym
    :return: space object in Pyrado
    """
    if isinstance(gym_space, gs.Box):
        return BoxSpace(gym_space.low, gym_space.high)
    if isinstance(gym_space, gs.Discrete):
        warn(
            "Guessing the conversion of a discrete OpenAI gym space. This feature is not tested. "
            "Rather use their control environments with continuous action spaces."
        )
        return DiscreteSpace(np.ones((gym_space.n, 1)))
    else:
        raise pyrado.TypeErr(msg=f"Unsupported space form gym {gym_space}")


class GymEnv(SimEnv, Serializable):
    """A Wrapper to use the classical control environments of OpenAI Gym like Pyrado environments"""

    name: str = "gym-cc"

    def __init__(self, env_name: str):
        """
        Constructor

        .. note::
            Pyrado only supports the classical control environments from OpenAI Gym.
            See https://github.com/openai/gym/tree/master/gym/envs/classic_control

        :param env_name: name of the OpenAI Gym environment, e.g. 'MountainCar-v0', 'CartPole-v1', 'Acrobot-v1',
                         'MountainCarContinuous-v0','Pendulum-v0'
        """
        Serializable._init(self, locals())

        # Initialize basic variables
        if env_name == "MountainCar-v0":
            dt = 0.02  # there is no dt in the source file
        elif env_name == "CartPole-v1":
            dt = 0.02
        elif env_name == "Acrobot-v1":
            dt = 0.2
        elif env_name == "MountainCarContinuous-v0":
            dt = 0.02  # there is no dt in the source file
        elif env_name == "Pendulum-v0":
            dt = 0.05
        elif env_name == "LunarLander-v2":
            dt = 0.02
        else:
            raise NotImplementedError(f"GymEnv does not wrap the environment {env_name}.")
        super().__init__(dt)

        # Create the gym environment
        self._gym_env = gym.envs.make(env_name)

        # Set the maximum number of time steps to 1000 if not given by the gym env
        self.max_steps = getattr(self._gym_env.spec, "max_episode_steps", 1000)

        # Create spaces compatible to Pyrado
        self._obs_space = _to_pyrado_space(self._gym_env.observation_space)
        self._act_space = _to_pyrado_space(self._gym_env.action_space)

    @property
    def state_space(self) -> Space:
        # Copy of obs_space since the OpenAI gym has no dedicated state space
        return self._obs_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def init_space(self) -> None:
        # OpenAI Gym environments do not have an init_space
        return None

    @property
    def act_space(self) -> Space:
        return self._act_space

    def _create_task(self, task_args: dict) -> None:
        return None

    @property
    def task(self):
        # Doesn't have any
        return None

    @property
    def domain_param(self) -> dict:
        # Doesn't have any
        return {}

    @domain_param.setter
    def domain_param(self, domain_param):
        # Ignore
        pass

    @classmethod
    def get_nominal_domain_param(cls):
        # Doesn't have any
        return {}

    def reset(self, init_state=None, domain_param=None):
        return self._gym_env.reset()

    def step(self, act) -> tuple:
        if isinstance(self.act_space, DiscreteSpace):
            act = act.astype(dtype=np.int64)  # PyTorch policies operate on doubles but discrete gym envs want integers
            act = act.item()  # discrete gym envs want integers or scalar arrays
        return self._gym_env.step(act)

    def render(self, mode: RenderMode = RenderMode(), render_step: int = 1):
        if mode.video:
            return self._gym_env.render()

    def close(self):
        return self._gym_env.close()
