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

import numpy as np
from init_args_serializer import Serializable
from rcsenv import JointLimitException, RcsSimEnv

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.empty import EmptySpace
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode


def to_pyrado_space(space) -> [BoxSpace, EmptySpace]:
    """
    Convert the box space implementation from RcsPySim to the one of Pyrado.

    :param space: a space from RcsPySim
    :return: a Pyrado `BoxSpace` or an Pyrado`EmptySpace` if `None` was given
    """
    if space is None:
        return EmptySpace
    return BoxSpace(space.min, space.max, labels=space.names)


class RcsSim(SimEnv, Serializable):
    """ Base class for RcsPySim environments. Uses Serializable to facilitate proper serialization. """

    def __init__(
        self,
        envType: str,
        task_args: dict,
        dt: float = 0.01,
        max_steps: int = pyrado.inf,
        init_state: np.ndarray = None,
        checkJointLimits: bool = False,
        joint_limit_penalty: float = -1e3,
        **kwargs,
    ):
        """
        Constructor

        .. note::
            The joint type (i.e. position or torque control) is set in the associated xml files in `RcsPySim/config`.

        :param envType: environment type name as defined on the C++ side
        :param task_args: arguments for the task construction
        :param dt: integration step size in seconds
        :param max_steps: max number of simulation time steps
        :param domain_param: initial domain param values
        :param init_state: initial state sampler can be a callable or one fixed state
        :param checkJointLimits: flags if the joint limits should be ignored or not passed to the C++ constructor
        :param joint_limit_penalty: cost returned on termination due to joint limits. This is a different from the
                                    state bounds since `RcsPySim` return an error when the joint limits are violated.
        :param kwargs: keyword arguments which are available for `RcsSim` on the C++ side. These arguments will not
                       be stored in the environment object, thus are saved e.g. when pickled.
        """
        Serializable._init(self, locals())

        # Initialize basic variables
        super().__init__(dt, max_steps)
        self._check_joint_limits = checkJointLimits

        # Create Rcs-based implementation (RcsSimEnv comes from the pybind11 module)
        self._sim = RcsSimEnv(dt=dt, envType=envType, checkJointLimits=self._check_joint_limits, **kwargs)

        # Setup the initial domain parameters
        self._domain_param = self._unadapt_domain_param(self._sim.domainParam)

        if joint_limit_penalty > 0:
            raise pyrado.ValueErr(given=joint_limit_penalty, le_constraint="0")
        self._joint_limit_penalty = joint_limit_penalty

        # Initial init state space is taken from C++
        self._init_space = to_pyrado_space(self._sim.initStateSpace)

        # By default, the state space is a subset of the observation space. Set this to customize in subclass.
        self.state_mask = None

        # Dummy initialization, must be set by the derived classes
        self.init_state = None
        self.task_args = task_args
        self._task = self._create_task(self.task_args)

    @property
    def state_space(self) -> Space:
        """ Derives the state space from the observation space using _state_from_obs or state_mask. """
        obs_space = self.obs_space
        # Check if _state_from_obs was overridden
        if self._state_from_obs.__func__ != RcsSim._state_from_obs:
            return BoxSpace(self._state_from_obs(obs_space.bound_lo), self._state_from_obs(obs_space.bound_up), None)
        # Check if there is a state mask
        if self.state_mask is not None:
            return obs_space.subspace(self.state_mask)
        # Identical to obs space
        return obs_space

    @property
    def obs_space(self) -> Space:
        return to_pyrado_space(self._sim.observationSpace)

    @property
    def init_space(self) -> Space:
        return to_pyrado_space(self._sim.initStateSpace)

    @property
    def act_space(self) -> Space:
        return to_pyrado_space(self._sim.actionSpace)

    @property
    def task(self) -> Task:
        return self._task

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        raise NotImplementedError

    @property
    def domain_param(self) -> dict:
        return self._unadapt_domain_param(self._sim.domainParam)

    @domain_param.setter
    def domain_param(self, domain_param: dict):
        if not isinstance(domain_param, dict):
            raise pyrado.TypeErr(given=domain_param, expected_type=dict)
        # Update the internal parameters. The New domain parameters will be applied on reset().
        self._domain_param.update(domain_param)

        # Update task
        self._task = self._create_task(self.task_args)

    @classmethod
    def get_nominal_domain_param(cls):
        """
        Get the nominal a.k.a. default domain parameters.

        .. note::
            It is highly recommended to have the same values as in the associated physics config file (p<NAME>.xml),
            since the nominal domain parameters are not set explicitly from Pyrado (only when randomizing).
        """
        raise NotImplementedError

    def _state_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Retrieve the system state from the observation. In most cases, the system state is a part of the observation.
        This function is to be used when the observations include additional information.
        The default implementation is based off `self.state_mask` which is set in sub-classes of `RcsSim`.

        :param obs: observation from the environment
        :return: state of the environment
        """
        if self.state_mask is not None:
            return obs[self.state_mask]
        return obs.copy()

    def _adapt_domain_param(self, params: dict) -> dict:
        """
        Changes the domain parameters before passing them to the Rcs simulation.
        One use case is for example the rolling friction coefficient which is usually given unit-less but the Vortex
        physics engine expects it to be multiplied with the body's curvature radius.

        :param params: domain parameters to adapt
        :return: adapted parameters
        """
        return params

    def _unadapt_domain_param(self, params: dict) -> dict:
        """
        Changes the domain parameters coming from to the Rcs simulation.

        .. note::
            This function is called from the constructor.

        :param params: domain parameters to revert the previously done adaptation
        :return: unadapted parameters
        """
        return params

    def _get_state(self, state_dict: dict):
        state_dict["domain_param"] = self.domain_param
        state_dict["init_state"] = self.init_state

    def _set_state(self, state_dict: dict, copying: bool = False):
        self.domain_param = state_dict["domain_param"]
        self.init_state = state_dict["init_state"]

    def _disturbance_generator(self) -> (np.ndarray, None):
        """ Provide an artificial disturbance. For example a force on a body in the physics simulation. """
        return None

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset time
        self._curr_step = 0

        # Reset the state
        if init_state is None:
            # Sample from the init state space
            init_state = self._init_space.sample_uniform()
        else:
            if not init_state.shape == self._init_space.shape:
                raise pyrado.ShapeErr(given=init_state, expected_match=self._init_space)

        # Reset the task
        self._task.reset(env_spec=self.spec)

        # Use stored domain parameters if not overwritten
        if domain_param is None:
            domain_param = self._domain_param

        # Forward to C++ implementation
        obs = self._sim.reset(domainParam=self._adapt_domain_param(domain_param), initState=init_state)
        self.state = self._state_from_obs(obs)

        return obs

    def step(self, act: np.ndarray) -> tuple:
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act = self.limit_act(act)

        # Get the disturbance to be applied on the Rcs side
        disturbance = self._disturbance_generator()

        # Dynamics are calculated in the Rcs simulation
        try:
            obs = self._sim.step(act, disturbance)
        except JointLimitException:
            # Joint limits exceeded! Return (obs, rew, done, info) directly after this failure.
            return self._sim.lastObservation, self._joint_limit_penalty, True, dict(t=self._curr_step * self._dt)

        self.state = self._state_from_obs(obs)  # only for the Python side

        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return obs, self._curr_rew, done, dict()

    def render(self, mode: RenderMode = RenderMode(text=True), render_step: int = 1):
        if self._curr_step % render_step == 0:
            # Call base class
            super().render(mode)

            # Forward to Rcs GUI
            if mode.video:
                self._sim.render()

    def save_config_xml(self, fileName: str):
        """
        Save environment configuration as xml file for use on the C++ side.

        :param fileName: output file name
        """
        self._sim.saveConfigXML(fileName)

    def get_body_position(self, bodyName: str, refFrameName: str, refBodyName: str) -> np.ndarray:
        """
        Get the position of a body in the simulators config graph.
        This function uses code coped from `Rcs` to transform the position depending on a refernce frame and/or body.

        :param bodyName: name of the body in the graph
        :param refFrameName: name of the reference frame, pass '' to use world coordinates
        :param refBodyName: name of the reference body, pass '' to use world coordinates
        :return: x,y,z positions in a reference frame coordinates relative to a reference bodies
        """
        return self._sim.getBodyPosition(bodyName, refFrameName, refBodyName)

    def get_body_extents(self, bodyName: str, shapeIdx: int = 0) -> np.ndarray:
        """
        Get the dimensions of a body in the simulators config graph.
        This function uses code coped from `Rcs` to transform the position depending on a refernce frame and/or body.

        .. note::
            Depending on the kind of shape (e.g. box, sphere, torus, ect.) the extends mean different things.

        :param bodyName: name of the body in the graph
        :param shapeIdx: index of the shape in the `Body` node, defaults to the first shape of the body
        :return: x,y,z positions in a reference frame coordinates relative to a reference bodies
        """
        return self._sim.getBodyExtents(bodyName, shapeIdx)
