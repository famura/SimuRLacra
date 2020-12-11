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
import numpy as np
import robcom_python as robcom
from abc import ABC, abstractmethod
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.barrett_wam import (
    init_qpos_des_4dof,
    init_qpos_des_7dof,
    act_space_wam_7dof,
    act_space_wam_4dof,
    wam_pgains,
    wam_dgains,
)
from pyrado.environments.real_base import RealEnv
from pyrado.spaces import BoxSpace
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt, completion_context, print_cbt_once


class WAMBallInCupReal(RealEnv, ABC, Serializable):
    """
    Abstract base class for the real Barrett WAM

    Uses robcom 2.0 and specifically robcom's ClosedLoopDirectControl process to execute a trajectory
    given by desired joint positions.

    The concrete control approach (step-based or episodic) is implemented by the sub class
    """

    name: str = "wam-bic"

    def __init__(
        self, dt: float = 1 / 500.0, max_steps: int = pyrado.inf, num_dof: int = 7, ip: [str, None] = "192.168.2.2"
    ):
        """
        Constructor

        :param dt: sampling time interval
        :param max_steps: maximum number of time steps
        :param num_dof: number of degrees of freedom (4 or 7), depending on which Barrett WAM setup being used
        :param ip: IP address of the PC controlling the Barrett WAM, pass `None` to skip connecting
        """
        Serializable._init(self, locals())

        # Make sure max_steps is reachable
        if not max_steps < pyrado.inf:
            raise pyrado.ValueErr(given=max_steps, given_name="max_steps", l_constraint=pyrado.inf)

        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Create the robcom client and connect to it. Use a Process to timeout if connection cannot be established
        self._connected = False
        self._client = robcom.Client()
        self._robot_group_name = "RIGHT_ARM"
        try:
            self._client.start(ip, 2013, 1000)  # ip address, port, timeout in ms
            self._connected = True
            print_cbt("Connected to the Barret WAM client.", "c", bright=True)
        except RuntimeError:
            print_cbt("Connection to the Barret WAM client failed.", "r", bright=True)
        self._jg = self._client.robot.get_group([self._robot_group_name])
        self._dc = None  # direct-control process
        self._t = None  # only needed for WAMBallInCupRealStepBased

        # Number of controlled joints (dof)
        self.num_dof = num_dof

        # Desired joint position for the initial state and indices of the joints the policy operates on
        if self.num_dof == 4:
            self.qpos_des_init = init_qpos_des_4dof
            self.idcs_act = [1, 3]
        elif self.num_dof == 7:
            self.qpos_des_init = init_qpos_des_7dof
            self.idcs_act = [1, 3, 5]
        else:
            raise pyrado.ValueErr(given=self.num_dof, eq_constraint="4 or 7")

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._create_spaces()

        # Initialize task
        self._task = self._create_task(dict())

        self.qpos_real = None
        self.qvel_real = None

    @property
    def state_space(self) -> Space:
        return self._state_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def act_space(self) -> Space:
        return self._act_space

    @property
    def task(self) -> Task:
        return self._task

    def _create_task(self, task_args: dict) -> Task:
        # The wrapped task acts as a dummy and carries the FinalRewTask
        return FinalRewTask(GoallessTask(self.spec, ZeroPerStepRewFcn()), mode=FinalRewMode(user_input=True))

    @abstractmethod
    def _create_spaces(self):
        """
        Create spaces based on the domain parameters.
        Should set the attributes `_state_space`, `_act_space`, and `_obs_space`.

        .. note::
            This function is called from the constructor.
        """
        raise NotImplementedError

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        if not self._connected:
            print_cbt("Not connected to Barret WAM client.", "r", bright=True)
            raise pyrado.ValueErr(given=self._connected, eq_constraint=True)

        # Create a direct control process to set the PD gains
        self._client.set(robcom.Streaming, 500.0)  # Hz
        dc = self._client.create(robcom.DirectControl, self._robot_group_name, "")
        dc.start()
        dc.groups.set(robcom.JointDesState.P_GAIN, wam_pgains[: self.num_dof].tolist())
        dc.groups.set(robcom.JointDesState.D_GAIN, wam_dgains[: self.num_dof].tolist())
        dc.send_updates()
        dc.stop()

        # Read and print the set gains to confirm that they were set correctly
        time.sleep(0.1)  # short active waiting because updates are sent in another thread
        pgains_des = self._jg.get_desired(robcom.JointDesState.P_GAIN)
        dgains_des = self._jg.get_desired(robcom.JointDesState.D_GAIN)
        print_cbt(f"Desired PD gains are set to: {pgains_des} \t {dgains_des}", color="g")

        # Create robcom GoTo process
        gt = self._client.create(robcom.Goto, self._robot_group_name, "")

        # Move to initial state within 5 seconds
        gt.add_step(5.0, self.qpos_des_init)

        # Start process and wait for completion
        with completion_context("Moving the Barret WAM to the initial position", color="c", bright=True):
            gt.start()
            gt.wait_for_completion()

        # Reset the task which also resets the reward function if necessary
        self._task.reset(env_spec=self.spec)

        # Reset time steps
        self._curr_step = 0

        # Reset real WAM trajectory container
        self.qpos_real = np.zeros((self.max_steps, self.num_dof))
        self.qvel_real = np.zeros((self.max_steps, self.num_dof))

        # Reset the control process as well as state and trajectory params
        input("Hit enter to continue.")
        self._reset()

        return self.observe(self.state)

    @abstractmethod
    def _reset(self):
        """
        Custom reset function depending on the subclass `WAMBallInCupRealEpisodic` or `WAMBallInCupRealStepBased`.
        Resets the state and the trajectory params, and initializes control type specific variables.
        """
        raise NotImplementedError

    def render(self, mode: RenderMode, render_step: int = 1):
        # Skip all rendering
        pass

    def close(self):
        # Don't close the connection to robcom manually, since this might cause SL to crash.
        # Closing the connection is finally handled by robcom
        pass


class WAMBallInCupRealEpisodic(WAMBallInCupReal, Serializable):
    """
    Class for the real Barrett WAM

    Uses robcom 2.0 and specifically robcom's ClosedLoopDirectControl process to execute a trajectory
    given by desired joint positions. The control process is only executed on the real system after `max_steps` has been
    reached to avoid possible latency, but at the same time mimic the usual step-based environment behavior.
    """

    def _create_spaces(self):
        # State space (normalized time, since we do not have a simulation)
        self._state_space = BoxSpace(np.array([0.0]), np.array([1.0]))

        # Action space (PD controller on joint positions and velocities)
        if self.num_dof == 4:
            self._act_space = act_space_wam_4dof
        elif self.num_dof == 7:
            self._act_space = act_space_wam_7dof

        # Observation space (normalized time)
        self._obs_space = BoxSpace(np.array([0.0]), np.array([1.0]), labels=["t"])

    def _reset(self):
        # Reset current step of the real robot
        self._curr_step_rr = 0

        # Reset state
        self.state = np.array([self._curr_step / self.max_steps])

        # Reset trajectory params
        self.qpos_des = np.tile(self.qpos_des_init, (self.max_steps, 1))
        self.qvel_des = np.zeros_like(self.qpos_des)

        # Create robcom direct-control process
        self._dc = self._client.create(robcom.ClosedLoopDirectControl, self._robot_group_name, "")

    def step(self, act: np.ndarray) -> tuple:
        if self._curr_step == 0:
            print_cbt("Pre-sampling policy...", "w")

        info = dict(t=self._curr_step * self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)  # always 0 for wam-bic-real

        # Limit the action
        act = self.limit_act(act)

        # The policy operates on specific indices self.idcs_act, i.e. joint 1 and 3 (and 5)
        self.qpos_des[self._curr_step, self.idcs_act] += act[: len(self.idcs_act)]
        self.qvel_des[self._curr_step, self.idcs_act] += act[len(self.idcs_act) :]

        # Update current step and state
        self._curr_step += 1
        self.state = np.array([self._curr_step / self.max_steps])

        # A GoallessTask only signals done when has_failed() is true, i.e. the the state is out of bounds
        done = self._task.is_done(self.state)  # always false for wam-bic-real

        # Only start execution of process when all desired poses have been sampled from the policy
        if self._curr_step >= self._max_steps:
            done = True  # exceeded max time steps
            with completion_context("Executing trajectory on Barret WAM", color="c", bright=True):
                self._dc.start(False, round(500 * self._dt), self._callback, ["POS", "VEL"], [], [])
                t_start = time.time()
                self._dc.wait_for_completion()
                t_stop = time.time()
            print_cbt(f"Execution took {t_stop - t_start:1.5f} s.", "g")

        # Add final reward if done
        if done:
            # Ask the user to enter the final reward
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

            # Stop robcom data streaming
            self._client.set(robcom.Streaming, False)

        return self.observe(self.state), self._curr_rew, done, info

    def _callback(self, jg, eg, data_provider):
        """
        This function is called from robcom's ClosedLoopDirectControl process as callback and should never be called manually

        :param jg: joint group
        :param eg: end-effector group
        :param data_provider: additional data stream
        """
        # Check if max_steps is reached
        if self._curr_step_rr >= self.max_steps:
            return True

        # Get current joint position and velocity for storing
        self.qpos_real[self._curr_step_rr] = np.array(jg.get(robcom.JointState.POS))
        self.qvel_real[self._curr_step_rr] = np.array(jg.get(robcom.JointState.VEL))

        # Set desired joint position and velocity
        dpos = self.qpos_des[self._curr_step_rr].tolist()
        dvel = self.qvel_des[self._curr_step_rr].tolist()
        jg.set(robcom.JointDesState.POS, dpos)
        jg.set(robcom.JointDesState.VEL, dvel)

        # Update current step at real robot
        self._curr_step_rr += 1

        return False


class WAMBallInCupRealStepBased(WAMBallInCupReal, Serializable):
    """
    Class for the real Barrett WAM

    Uses robcom 2.0 and specifically robcom's ClosedLoopDirectControl process to execute a trajectory
    given by desired joint positions. The control process is running in a separate thread and
    executed on the real system simultaneous to the step function calls.
    """

    def _create_spaces(self):
        # State space (joint positions and velocities)
        state_shape = (2 * self.num_dof,)
        state_up, state_lo = np.full(state_shape, pyrado.inf), np.full(state_shape, -pyrado.inf)
        self._state_space = BoxSpace(state_lo, state_up)

        # Action space (PD controller on joint positions and velocities)
        if self.num_dof == 4:
            self._act_space = act_space_wam_4dof
        elif self.num_dof == 7:
            self._act_space = act_space_wam_7dof

        # Observation space (normalized time)
        self._obs_space = BoxSpace(np.array([0.0]), np.array([1.0]), labels=["t"])

    def _reset(self):
        # Get the robot access manager, to control that synchronized data is received
        self._ram = robcom.RobotAccessManager()

        # Reset desired positions and velocities
        self._qpos_des = self.qpos_des_init.copy()
        self._qvel_des = np.zeros_like(self.qpos_des_init)

        # Create robcom direct-control process
        self._dc = self._client.create(robcom.DirectControl, self._robot_group_name, "")

        # Get current joint state
        self.state = np.concatenate(self._get_joint_state())

        # Set the time for the busy waiting sleep call in step()
        self._t = time.time()

    def _get_joint_state(self):
        """
        Use robcom's streaming to get the current joint state

        :return: joint positions, joint velocities
        """
        self._ram.lock()
        qpos = self._jg.get(robcom.JointState.POS)
        qvel = self._jg.get(robcom.JointState.VEL)
        self._ram.unlock()

        return qpos, qvel

    def step(self, act: np.ndarray) -> tuple:
        # Start robcom direct-control process
        if self._curr_step == 0:
            print_cbt("Executing trajectory on Barret WAM", color="c", bright=True)
            self._dc.start()

        info = dict(t=self._curr_step * self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)  # always 0 for wam-bic-real

        # Limit the action
        act = self.limit_act(act)

        # The policy operates on specific indices `self.idcs_act`, i.e. joint 1 and 3 (and 5)
        self._qpos_des[self.idcs_act] = self.qpos_des_init[self.idcs_act] + act[: len(self.idcs_act)]
        self._qvel_des[self.idcs_act] = act[len(self.idcs_act) :]

        # Send desired positions and velocities to robcom
        self._dc.groups.set(robcom.JointDesState.POS, self._qpos_des)
        self._dc.groups.set(robcom.JointDesState.VEL, self._qvel_des)
        self._dc.send_updates()

        # Sleep to keep the frequency
        to_sleep = self._dt - (time.time() - self._t)
        if to_sleep > 0.0:
            time.sleep(to_sleep)
        else:
            print_cbt_once("The step call was too slow for the control frequency", color="y")
        self._t = time.time()

        # Get current joint angles and angular velocities
        qpos, qvel = self._get_joint_state()
        self.qpos_real[self._curr_step] = qpos
        self.qvel_real[self._curr_step] = qvel
        self.state = np.concatenate([qpos, qvel])

        # Update current step and state
        self._curr_step += 1

        # A GoallessTask only signals done when has_failed() is true, i.e. the the state is out of bounds
        done = self._task.is_done(self.state)  # always false for wam-bic-real

        # Check if exceeded max time steps
        if self._curr_step >= self._max_steps:
            done = True

        # Add final reward if done
        if done:
            # Ask the user to enter the final reward
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

            # Stop robcom direct-control process
            self._dc.stop()

            # Stop robcom data streaming
            self._client.set(robcom.Streaming, False)

        return self.observe(self.state), self._curr_rew, done, info

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.array([self._curr_step / self.max_steps])
