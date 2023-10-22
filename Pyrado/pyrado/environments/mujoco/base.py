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

from abc import ABC, abstractmethod
from copy import deepcopy
from math import floor
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


class MujocoSimEnv(SimEnv, ABC, Serializable):
    """
    Base class for MuJoCo environments.
    Uses Serializable to facilitate proper serialization.

    .. seealso::
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
    """

    def __init__(
        self,
        model_path: str,
        frame_skip: int = 1,
        dt: Optional[float] = None,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
    ):
        """
        Constructor

        :param model_path: path to the MuJoCo xml model config file
        :param frame_skip: number of simulation frames for which the same action is held, results in a multiplier of
                           the time step size `dt`
        :param dt: by default the time step size is the one from the mujoco config file multiplied by the number of
                   frame skips (legacy from OpenAI environments). By passing an explicit `dt` value, this can be
                   overwritten. Possible use case if if you know that you recorded a trajectory with a specific `dt`.
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction, e.g `dict(fwd_rew_weight=1.)`
        """
        Serializable._init(self, locals())

        # Initialize
        self.model_path = model_path
        self._domain_param = self.get_nominal_domain_param()
        if dt is None:
            # Specify the time step size as a multiple of MuJoCo's simulation time step size
            self.frame_skip = frame_skip
        else:
            # Specify the time step size explicitly
            with open(self.model_path, mode="r") as file_raw:
                xml_model_temp = file_raw.read()
            xml_model_temp = self._adapt_model_file(xml_model_temp, self.domain_param)
            # Create a dummy model to extract the solver's time step size
            model_tmp = mujoco.MjModel.from_xml_path(xml_model_temp)
            frame_skip = dt / model_tmp.opt.timestep
            if frame_skip.is_integer():
                self.frame_skip = int(frame_skip)
            elif dt > model_tmp.opt.timestep:
                print_cbt(
                    f"The desired time step size is {dt} s, but solver's time step size in the MuJoCo config file is "
                    f"{model_tmp.opt.timestep} s. Thus, frame_skip is rounded down to {floor(frame_skip)}.",
                    "y",
                )
                self.frame_skip = floor(frame_skip)
            else:
                # The number of skipped frames must be >= 1
                pyrado.ValueErr(given=dt, ge_constraint=model_tmp.opt.timestep)

        # Creat the MuJoCo model
        with open(self.model_path, mode="r") as file_raw:
            # Save raw (with placeholders) XML-file as attribute since we need it for resetting the domain params
            self.xml_model_template = file_raw.read()
        self._create_mujoco_model()

        # Call SimEnv's constructor
        super().__init__(dt=self.model.opt.timestep * self.frame_skip, max_steps=max_steps)

        # Memorize the initial states of the model from the xml (for fixed init space or later reset)
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # Initialize space (to be overwritten in constructor of subclasses)
        self._init_space = None

        # Create task
        if not (isinstance(task_args, dict) or task_args is None):
            raise pyrado.TypeErr(given=task_args, expected_type=dict)
        self.task_args = dict() if task_args is None else task_args
        self._task = self._create_task(self.task_args)

        # Visualization
        self.camera_config = dict()
        self.viewer = None
        self._curr_act = np.zeros(self.act_space.shape)

    @property
    @abstractmethod
    def state_space(self) -> Space:
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Space:
        raise NotImplementedError

    @property
    @abstractmethod
    def act_space(self) -> Space:
        raise NotImplementedError

    @property
    def init_space(self) -> Space:
        return self._init_space

    @init_space.setter
    def init_space(self, space: Space):
        if not isinstance(space, Space):
            raise pyrado.TypeErr(given=space, expected_type=Space)
        self._init_space = space

    @property
    def task(self) -> Task:
        return self._task

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        raise NotImplementedError

    @property
    def domain_param(self) -> dict:
        return deepcopy(self._domain_param)

    @domain_param.setter
    def domain_param(self, domain_param: dict):
        if not isinstance(domain_param, dict):
            raise pyrado.TypeErr(given=domain_param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(domain_param)

        # Update MuJoCo model
        self._create_mujoco_model()

        self.viewer = None

        # Update task
        self._task = self._create_task(self.task_args)

    def _adapt_model_file(self, xml_model: str, domain_param: dict) -> str:
        """
        Changes the model's XML-file given the current domain parameters before constructing the MuJoCo simulation.
        One use case is for example the cup_scale for the `WAMBallInCupSim` where multiple values in the model's
        XML-file are changed based on one domain parameter.

        .. note::
            It is mandatory to call this function in case you modified the mxl config file with tags like `[DP_NAME]`.

        :param xml_model: parsed model file
        :param domain_param: copy of the environments domain parameters
        :return: adapted model file where the placeholders are filled with numerical values
        """
        # The mesh dir is not resolved when later passed as a string, thus we do it manually
        xml_model = xml_model.replace("[ASSETS_DIR]", pyrado.MUJOCO_ASSETS_DIR)

        # Replace all occurrences of the domain parameter placeholder with its value
        for key, value in domain_param.items():
            xml_model = xml_model.replace(f"[{key}]", str(value))

        return xml_model

    @abstractmethod
    def _mujoco_step(self, act: np.ndarray) -> dict:
        """
        Apply the given action to the MuJoCo simulation. This executes one step of the physics simulation.

        :param act: action
        :return: dictionary with optional information from MuJoCo
        """

    def _create_mujoco_model(self):
        """
        Called to update the MuJoCo model by rewriting and reloading the XML file.

        .. note::
            This function is called from the constructor and from the domain parameter setter.
        """
        xml_model = self.xml_model_template  # don't change the template
        xml_model = self._adapt_model_file(xml_model, self.domain_param)

        # Create MuJoCo model from parsed XML file
        self.model = mujoco.MjModel.from_xml_string(xml_model)
        self.data = mujoco.MjData(self.model)

    def configure_viewer(self):
        """Configure the camera when the viewer is initialized. You need to set `self.camera_config` before."""
        # Render a fog around the scene by default
        if self.camera_config.pop("render_fog", True):
            self.viewer.scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1

        # Parse all other options
        for key, value in self.camera_config.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset time
        self._curr_step = 0

        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param

        # Sample or set the initial simulation state
        if init_state is None:
            # Sample init state from init state space
            init_state = self.init_space.sample_uniform()
        elif not isinstance(init_state, np.ndarray):
            # Make sure init state is a numpy array
            try:
                init_state = np.asarray(init_state)
            except Exception:
                raise pyrado.TypeErr(given=init_state, expected_type=np.ndarray)
        if not self.init_space.contains(init_state, verbose=True):
            raise pyrado.ValueErr(msg="The init state must be within init state space!")

        # Update the state attribute
        self.state = init_state.copy()

        # Reset the task which also resets the reward function if necessary
        self._task.reset(env_spec=self.spec, init_state=init_state.copy())

        # Reset MuJoCo simulation model (only reset the joint configuration)
        mujoco.mj_resetData(self.model, self.data)
        old_state = self.data
        nq = self.model.nq
        nv = self.model.nv
        if not init_state[:nq].shape == old_state.qpos.shape:  # check joint positions dimension
            raise pyrado.ShapeErr(given=init_state[:nq], expected_match=old_state.qpos)
        # Exclude everything that is appended to the state (at the end), e.g. the ball position for WAMBallInCupSim
        if not init_state[nq : nq + nv].shape == old_state.qvel.shape:  # check joint velocities dimension
            raise pyrado.ShapeErr(given=init_state[nq : nq + nv], expected_match=old_state.qvel)
        self.data.qpos[:] = np.copy(init_state[:nq])
        self.data.qvel[:] = np.copy(init_state[nq : nq + nv])
        mujoco.mj_forward(self.model, self.data)

        # Return an observation
        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self.task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act = self.limit_act(act)
        self._curr_act = act  # just for the render function

        # Apply the action and simulate the resulting dynamics
        info = self._mujoco_step(act)
        self._curr_step += 1

        # Check if the environment is done due to a failure within the mujoco simulation (e.g. bad inputs)
        mjsim_done = info.get("failed", False)

        # Check if the task is done
        task_done = self._task.is_done(self.state)

        # Handle done case
        done = mjsim_done or task_done
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode = RenderMode(), render_step: int = 1):
        if self._curr_step % render_step == 0:
            # Call base class
            super().render(mode)

            # Print to console
            if mode.text:
                print(
                    f"step: {self._curr_step:4d}  |  r_t: {self._curr_rew: 1.3f}  |  a_t: {self._curr_act}  |  s_t+1: {self.state}"
                )

            # Forward to MuJoCo viewer
            if mode.video:
                if self.viewer is None:
                    # Create viewer if not existent (see 'human' mode of OpenAI Gym's MujocoEnv)
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

                    self.configure_viewer()
                self.viewer.sync()
