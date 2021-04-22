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

from abc import abstractmethod
from typing import Iterable, Optional, Union

import numpy as np
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode


class EnvWrapper(Env, Serializable):
    """ Base for all environment wrappers. Delegates all environment methods to the wrapped environment. """

    def __init__(self, wrapped_env: Env):
        """
        Constructor

        :param wrapped_env: environment to wrap
        """
        if not isinstance(wrapped_env, Env):
            raise pyrado.TypeErr(given=wrapped_env, expected_type=Env)

        Serializable._init(self, locals())

        self._wrapped_env = wrapped_env

    @property
    def name(self) -> str:
        """ Get the wrapped environment's abbreviated name. """
        return self._wrapped_env.name

    @property
    def wrapped_env(self) -> Env:
        """ Get the wrapped environment of this wrapper. """
        return self._wrapped_env

    @property
    def state_space(self) -> Space:
        return self._wrapped_env.state_space

    @property
    def obs_space(self) -> Space:
        return self._wrapped_env.obs_space

    @property
    def act_space(self) -> Space:
        return self._wrapped_env.act_space

    @property
    def init_space(self) -> Space:
        """ Get the initial state space if it exists. Forwards to the wrapped environment. """
        if isinstance(self._wrapped_env, (SimEnv, EnvWrapper)):
            return self._wrapped_env.init_space
        else:
            raise NotImplementedError

    @init_space.setter
    def init_space(self, space: Space):
        """ Set the initial state space if it exists. Forwards to the wrapped environment. """
        if isinstance(self._wrapped_env, (SimEnv, EnvWrapper)):
            self._wrapped_env.init_space = space
        else:
            raise NotImplementedError

    @property
    def dt(self):
        return self._wrapped_env.dt

    @dt.setter
    def dt(self, dt: float):
        self._wrapped_env.dt = dt

    @property
    def curr_step(self) -> int:
        return self._wrapped_env.curr_step

    @property
    def max_steps(self) -> Union[int, float]:
        return self._wrapped_env.max_steps

    @max_steps.setter
    def max_steps(self, num_steps: int):
        self._wrapped_env.max_steps = num_steps

    @property
    def state(self) -> np.ndarray:
        """ Get the state of the wrapped environment. """
        return self._wrapped_env.state.copy()

    @state.setter
    def state(self, state: np.ndarray):
        """ Set the state of the wrapped environment. """
        if not isinstance(state, np.ndarray):
            raise pyrado.TypeErr(given=state, expected_type=np.ndarray)
        if not state.shape == self._wrapped_env.state.shape:
            raise pyrado.ShapeErr(given=state, expected_match=self._wrapped_env.state)
        self._wrapped_env.state = state

    def _create_task(self, task_args: dict) -> Task:
        return self._wrapped_env._create_task(task_args)

    @property
    def task(self) -> Task:
        return self._wrapped_env.task

    @property
    def domain_param(self) -> dict:
        """
        These are the environment's domain parameters, which are synonymous to the parameters used by the simulator to
        run the physics simulation (e.g., masses, extents, or friction coefficients). The property domain_param includes
        all parameters that can be perturbed a.k.a. randomized, but there might also be additional parameters.
        """
        if not isinstance(self._wrapped_env, (SimEnv, EnvWrapper)):
            raise pyrado.TypeErr(given=self._wrapped_env, expected_type=(SimEnv, EnvWrapper))
        else:
            param = self._wrapped_env.domain_param
            self._set_wrapper_domain_param(param)
            return param

    @domain_param.setter
    def domain_param(self, domain_param: dict):
        """
        Set the environment's domain parameters. The changes are only applied at the next call of the reset function.

        :param domain_param: new domain parameter set
        """
        self._get_wrapper_domain_param(domain_param)
        self._wrapped_env.domain_param = domain_param

    def get_nominal_domain_param(self) -> dict:
        """
        Get the nominal a.k.a. default domain parameters.

        .. note::
            This function is used to check which domain parameters exist.
        """
        if not isinstance(self._wrapped_env, (SimEnv, EnvWrapper)):
            raise pyrado.TypeErr(given=self._wrapped_env, expected_type=(SimEnv, EnvWrapper))
        else:
            return self._wrapped_env.get_nominal_domain_param()

    @property
    def supported_domain_param(self) -> Iterable:
        """
        Get an iterable of all supported domain parameters.
        The default implementation takes the keys of `get_nominal_domain_param()`.
        The domain parameters are automatically stored in attributes prefixed with '_'.
        """
        return self._wrapped_env.supported_domain_param

    def forward(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        """
        Recursively go though the stack of wrappers and try to apply the forward transformation.
        This assumes that there is only one.

        :param value: domain parameter value in the original space
        :return: domain parameter value in the transformed space
        """
        forward_fcn = getattr(self._wrapped_env, "forward", None)
        if callable(forward_fcn):
            return forward_fcn(value)
        else:
            # Arrived at the inner env, no transformation found
            return value

    def inverse(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        """
        Recursively go though the stack of wrappers and try to apply the inverse transformation.
        This assumes that there is only one.

        :param value: domain parameter value in the transformed space
        :return: domain parameter value in the original space
        """
        inverse_fcn = getattr(self._wrapped_env, "inverse", None)
        if callable(inverse_fcn):
            return inverse_fcn(value)
        else:
            # Arrived at the inner env, no transformation found
            return value

    @property
    def randomizer(self) -> Optional[DomainRandomizer]:
        """ Get the wrapped environment's domain randomizer. """
        return getattr(self._wrapped_env, "randomizer", None)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        """
        Reset the environment to its initial state and optionally set different domain parameters.

        :param init_state: set explicit initial state if not None
        :param domain_param: set explicit domain parameters if not None
        :return obs: initial observation of the state.
        """
        if domain_param is not None:
            self._get_wrapper_domain_param(domain_param)
        return self._wrapped_env.reset(init_state, domain_param)

    def step(self, act: np.ndarray) -> tuple:
        """
        Perform one time step of the simulation. When a terminal condition is met, the reset function is called.

        :param act: action to be taken in the step
        :return tuple of obs, reward, done, and info:
                obs : current observation of the environment
                reward: reward depending on the selected reward function
                done: indicates whether the episode has ended
                env_info: contains diagnostic information about the environment
        """
        return self._wrapped_env.step(act)

    def render(self, mode: RenderMode, render_step: int = 1):
        self._wrapped_env.render(mode, render_step)

    def close(self):
        return self._wrapped_env.close()

    def _get_wrapper_domain_param(self, param: dict):
        """ Called by the domain_param setter. Use to load wrapper-specific params. Does nothing by default. """
        pass

    def _set_wrapper_domain_param(self, param: dict):
        """ Called by the domain_param getter. Use to store wrapper-specific params. Does nothing by default. """
        pass


class EnvWrapperAct(EnvWrapper):
    """
    Base class for environment wrappers modifying the action.
    Override _process_action to pass a modified action vector to the wrapped environment.
    If necessary, you should also override _process_action_space to report the correct one.
    """

    @abstractmethod
    def _process_act(self, act: np.ndarray):
        """
        Return the modified action vector to be passed to the wrapped environment.

        :param act: action vector (should not be modified in place)
        :return: changed action vector
        """
        raise NotImplementedError

    def _process_act_space(self, space: Space):
        """
        Return the modified action space. Override if the operation defined in _process_action affects
        shape or bounds of the action vector.
        :param space: inner env action space
        :return: action space to report for this env
        """
        return space

    def step(self, act: np.ndarray) -> tuple:
        # Modify action
        mod_act = self._process_act(act)

        # Delegate to base/wrapped
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperObs
        return super().step(mod_act)

    @property
    def act_space(self) -> Space:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperObs
        return self._process_act_space(super().act_space)


class EnvWrapperObs(EnvWrapper):
    """
    Base class for environment wrappers modifying the observation.
    Override _process_obs to pass a modified observation vector to the wrapped environment.
    If necessary, you should also override _process_obs_space to report the correct one.
    """

    @abstractmethod
    def _process_obs(self, obs: np.ndarray):
        """
        Return the modified observation vector to be returned from this environment.

        :param obs: observation from the inner environment
        :return: changed observation vector
        """
        raise NotImplementedError

    def _process_obs_space(self, space: Space) -> Space:
        """
        Return the modified observation space.
        Override if the operation defined in _process_obs affects shape or bounds of the observation vector.
        :param space: inner env observation space
        :return: action space to report for this env
        """
        return space

    @property
    def obs_space(self) -> Space:
        # Process space
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        return self._process_obs_space(super().obs_space)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        init_obs = super().reset(init_state, domain_param)

        # Return processed observation
        return self._process_obs(init_obs)

    def step(self, act: np.ndarray) -> tuple:
        # Step inner environment
        # By not using _wrapped_env directly, we can mix this class with EnvWrapperAct
        obs, rew, done, info = super().step(act)

        # Return processed observation
        return self._process_obs(obs), rew, done, info
