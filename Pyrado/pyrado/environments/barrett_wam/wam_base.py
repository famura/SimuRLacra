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

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import robcom_python as robcom  # pylint: disable=import-error

import pyrado
from pyrado.environments.barrett_wam import (
    init_qpos_des_4dof,
    init_qpos_des_7dof,
    wam_dgains_4dof,
    wam_dgains_7dof,
    wam_pgains_4dof,
    wam_pgains_7dof,
)
from pyrado.environments.real_base import RealEnv
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import completion_context, print_cbt


class WAMReal(RealEnv, ABC):
    """
    Abstract base class for the real Barrett WAM

    Uses robcom 2.0 and specifically robcom's `ClosedLoopDirectControl` process to execute a trajectory
    given by desired joint positions.

    The concrete control approach (step-based or episodic) is implemented by the subclass.
    """

    name: str = "wam"

    def __init__(
        self,
        num_dof: int,
        max_steps: int,
        dt: float = 1 / 500.0,
        ip: Optional[str] = "192.168.2.2",
    ):
        """
        Constructor

        :param num_dof: number of degrees of freedom (4 or 7), depending on which Barrett WAM setup being used
        :param max_steps: maximum number of time steps
        :param dt: sampling time interval, changing this value is highly discouraged
        :param ip: IP address of the PC controlling the Barrett WAM, pass `None` to skip connecting
        """
        # Make sure max_steps is reachable
        if not max_steps < pyrado.inf:
            raise pyrado.ValueErr(given=max_steps, given_name="max_steps", l_constraint=pyrado.inf)

        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Create the robcom client and connect to it. Use a Process to timeout if connection cannot be established.
        self._connected = False
        self._client = robcom.Client()
        self._robot_group_name = "RIGHT_ARM"
        try:
            self._client.start(ip, 2013, 1000)  # ip address, port, timeout in ms
            self._connected = True
            print_cbt("Connected to the Barret WAM client.", "c", bright=True)
        except RuntimeError:
            print_cbt("Connection to the Barret WAM client failed!", "r", bright=True)
        self._jg = self._client.robot.get_group([self._robot_group_name])
        self._dc = None  # direct-control process
        self._t = None  # only needed for WAMBallInCupRealStepBased

        # Desired joint position for the initial state and indices of the joints the policy operates on
        self._num_dof = num_dof
        if self._num_dof == 4:
            self._qpos_des_init = init_qpos_des_4dof
            self._idcs_act = [0, 1, 2, 3]  # use all joints by default
        elif self._num_dof == 7:
            self._qpos_des_init = init_qpos_des_7dof
            self._idcs_act = [0, 1, 2, 3, 4, 5, 6]  # use all joints by default
        else:
            raise pyrado.ValueErr(given=self._num_dof, eq_constraint="4 or 7")

        # Initialize task
        self._task = self._create_task(task_args=dict())

        # Trajectory containers (are set in reset())
        self.qpos_real = None
        self.qvel_real = None
        self.qpos_des = None
        self.qvel_des = None

    @property
    def num_dof(self) -> int:
        """Get the number of degrees of freedom."""
        return self._num_dof

    @property
    @abstractmethod
    def state_space(self) -> Space:
        """Get the state space."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Space:
        """Get the observation space."""
        raise NotImplementedError

    @property
    @abstractmethod
    def act_space(self) -> Space:
        """Get the action space."""
        raise NotImplementedError

    @property
    def task(self) -> Task:
        return self._task

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        if not self._connected:
            print_cbt("Not connected to Barret WAM client.", "r", bright=True)
            raise pyrado.ValueErr(given=self._connected, eq_constraint=True)

        # Create a direct control process to set the PD gains
        self._client.set(robcom.Streaming, 500.0)  # Hz
        dc = self._client.create(robcom.DirectControl, self._robot_group_name, "")
        dc.start()
        if self.num_dof == 4:
            dc.groups.set(robcom.JointDesState.P_GAIN, wam_pgains_4dof.tolist())
            dc.groups.set(robcom.JointDesState.D_GAIN, wam_dgains_4dof.tolist())
        else:
            dc.groups.set(robcom.JointDesState.P_GAIN, wam_pgains_7dof.tolist())
            dc.groups.set(robcom.JointDesState.D_GAIN, wam_dgains_7dof.tolist())
        dc.send_updates()
        dc.stop()

        # Read and print the set gains to confirm that they were set correctly
        time.sleep(0.1)  # short active waiting because updates are sent in another thread
        pgains_des = self._jg.get_desired(robcom.JointDesState.P_GAIN)
        dgains_des = self._jg.get_desired(robcom.JointDesState.D_GAIN)
        print_cbt(f"Desired P-gains have been set to {pgains_des}", color="g")
        print_cbt(f"Desired D-gains have been set to {dgains_des}", color="g")

        # Create robcom GoTo process
        gt = self._client.create(robcom.Goto, self._robot_group_name, "")

        # Move to initial state within 4 seconds
        gt.add_step(4.0, self._qpos_des_init)

        # Start process and wait for completion
        with completion_context("Moving the Barret WAM to the initial position", color="c", bright=True):
            gt.start()
            gt.wait_for_completion()

        # Reset the task which also resets the reward function if necessary
        self._task.reset(env_spec=self.spec)

        # Reset time steps
        self._curr_step = 0

        # Reset trajectory containers
        self.qpos_real = np.zeros((self.max_steps, self._num_dof))
        self.qvel_real = np.zeros((self.max_steps, self._num_dof))

        # return self.observe(self.state)
        return None  # TODO

    def render(self, mode: RenderMode = None, render_step: int = 1):
        # Skip all rendering
        pass

    def close(self):
        # Don't close the connection to robcom manually, since this might cause SL to crash.
        # Closing the connection is finally handled by robcom
        pass
