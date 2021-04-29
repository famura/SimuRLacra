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
from typing import Optional

import numpy as np
import robcom_python as robcom

import pyrado
from pyrado.environments.barrett_wam import (
    act_space_jsc_4dof,
    act_space_jsc_7dof,
    wam_q_limits_lo_7dof,
    wam_q_limits_up_7dof,
    wam_qd_limits_lo_7dof,
    wam_qd_limits_up_7dof,
)
from pyrado.environments.barrett_wam.wam_base import WAMReal
from pyrado.spaces import BoxSpace
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn
from pyrado.utils.input_output import completion_context, print_cbt


class WAMJointSpaceCtrlRealEpisodic(WAMReal):
    """
    Class for the real Barrett WAM, controlled by a PD controller using an episodic policy.

    Uses robcom 2.0 and specifically robcom's ClosedLoopDirectControl` process to execute a trajectory
    given by desired joint positions. The control process is only executed on the real system after `max_steps` has been
    reached to avoid possible latency, but at the same time mimic the usual step-based environment behavior.
    """

    name: str = "wam-jsc"

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
        :param dt: sampling time interval
        :param ip: IP address of the PC controlling the Barrett WAM, pass `None` to skip connecting
        """
        # Call WAMReal's constructor
        super().__init__(dt=dt, max_steps=max_steps, num_dof=num_dof, ip=ip)

        self._curr_step_rr = None

    @property
    def state_space(self) -> Space:
        # Normalized time
        return BoxSpace(np.array([0.0]), np.array([1.0]), labels=["t"])

    @property
    def obs_space(self) -> Space:
        lo = np.concatenate([wam_q_limits_lo_7dof[: self._num_dof], wam_qd_limits_lo_7dof[: self._num_dof]])
        up = np.concatenate([wam_q_limits_up_7dof[: self._num_dof], wam_qd_limits_up_7dof[: self._num_dof]])
        return BoxSpace(bound_lo=lo, bound_up=up)

    @property
    def act_space(self) -> Space:
        # Running a PD controller on joint positions and velocities
        return act_space_jsc_7dof if self._num_dof == 7 else act_space_jsc_4dof

    def _create_task(self, task_args: dict) -> Task:
        # Dummy task
        return GoallessTask(self.spec, ZeroPerStepRewFcn())

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Call WAMReal's reset
        super().reset(init_state, domain_param)

        # Reset current step of the real robot
        self._curr_step_rr = 0

        # Reset state
        self.state = np.array([self._curr_step / self.max_steps])

        # Reset trajectory params
        self.qpos_des = np.tile(self._qpos_des_init, (self.max_steps, 1))
        self.qvel_des = np.zeros_like(self.qpos_des)

        # Create robcom direct-control process
        self._dc = self._client.create(robcom.ClosedLoopDirectControl, self._robot_group_name, "")

        input("Hit enter to continue.")
        return self.observe(self.state)

    def step(self, act: np.ndarray) -> tuple:
        if self._curr_step == 0:
            print_cbt("Pre-sampling policy...", "w")

        info = dict(act_raw=act.copy())

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)  # always 0 for wam-bic-real

        # Limit the action
        act = self.limit_act(act)

        # The policy operates on specific indices self._idcs_act, i.e. joint 1 and 3 (and 5)
        self.qpos_des[self._curr_step, self._idcs_act] += act[: len(self._idcs_act)]
        self.qvel_des[self._curr_step, self._idcs_act] += act[len(self._idcs_act) :]

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


class WAMJointSpaceCtrlRealStepBased(WAMReal):
    """
    Class for the real Barrett WAM, controlled by a PD controller using an step-based policy.

    Uses robcom 2.0 and specifically robcom's `CosedLoopDirectControl` process to execute a trajectory
    given by desired joint positions. The control process is running in a separate thread and is executed on the real
    system simultaneous to the step function calls. Includes the option to observe ball and cup using OptiTrack.
    """

    name: str = "wam-jsc"

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
        :param dt: sampling time interval
        :param ip: IP address of the PC controlling the Barrett WAM, pass `None` to skip connecting
        """
        # Call WAMReal's constructor
        super().__init__(dt=dt, max_steps=max_steps, num_dof=num_dof, ip=ip)

        self._cnt_too_slow = None

    @property
    def state_space(self) -> Space:
        lo = np.concatenate([wam_q_limits_lo_7dof[: self._num_dof], wam_qd_limits_lo_7dof[: self._num_dof]])
        up = np.concatenate([wam_q_limits_up_7dof[: self._num_dof], wam_qd_limits_up_7dof[: self._num_dof]])
        return BoxSpace(
            bound_lo=lo,
            bound_up=up,
            labels=[f"q_{i}" for i in range(1, 8)] + [f"qd_{i}" for i in range(1, 8)],
        )

    @property
    def obs_space(self) -> Space:
        return self.state_space

    @property
    def act_space(self) -> Space:
        # Running a PD controller on joint positions and velocities
        return act_space_jsc_7dof if self._num_dof == 7 else act_space_jsc_4dof

    def _create_task(self, task_args: dict) -> Task:
        # Dummy task
        return GoallessTask(self.spec, ZeroPerStepRewFcn())

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Call WAMReal's reset
        super().reset(init_state, domain_param)

        # Reset desired positions and velocities
        self.qpos_des = self._qpos_des_init.copy()
        self.qvel_des = np.zeros_like(self._qpos_des_init)

        # Create robcom direct-control process
        self._dc = self._client.create(robcom.DirectControl, self._robot_group_name, "")

        # Get current joint state
        self.state = np.concatenate(self._get_joint_state())

        # Set the time for the busy waiting sleep call in step()
        self._t = time.time()
        self._cnt_too_slow = 0

        input("Hit enter to continue.")
        return self.observe(self.state)

    def _get_joint_state(self):
        """
        Use robcom's streaming to get the current joint state

        :return: joint positions, joint velocities
        """
        qpos = self._jg.get(robcom.JointState.POS)
        qvel = self._jg.get(robcom.JointState.VEL)

        return qpos, qvel

    def step(self, act: np.ndarray) -> tuple:
        # Start robcom direct-control process
        if self._curr_step == 0:
            print_cbt("Executing trajectory on Barret WAM", color="c", bright=True)
            self._dc.start()

        info = dict(act_raw=act.copy())

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)  # always 0 for wam-bic-real

        # Limit the action
        act = self.limit_act(act)

        # The policy operates on specific indices self._idcs_act, i.e. joint 1 and 3 (and 5)
        self.qpos_des[self._idcs_act] = self._qpos_des_init[self._idcs_act] + act[: len(self._idcs_act)]
        self.qvel_des[self._idcs_act] = act[len(self._idcs_act) :]

        # Send desired positions and velocities to robcom
        self._dc.groups.set(robcom.JointDesState.POS, self.qpos_des)
        self._dc.groups.set(robcom.JointDesState.VEL, self.qvel_des)
        self._dc.send_updates()

        # Sleep to keep the frequency
        to_sleep = self._dt - (time.time() - self._t)
        if to_sleep > 0.0:
            time.sleep(to_sleep)
        else:
            self._cnt_too_slow += 1
        self._t = time.time()

        # Get current joint angles and angular velocities
        qpos, qvel = self._get_joint_state()
        self.state = np.concatenate([qpos, qvel])
        self.qpos_real[self._curr_step] = qpos
        self.qvel_real[self._curr_step] = qvel

        # Update current step and state
        self._curr_step += 1

        # A GoallessTask only signals done when has_failed() is true, i.e. the the state is out of bounds
        done = self._task.is_done(self.state)

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

            print_cbt(
                f"The step call was too slow for the control frequency {self._cnt_too_slow} out of "
                f"{self._curr_step} times.",
                color="y",
            )

        return self.observe(self.state), self._curr_rew, done, info
