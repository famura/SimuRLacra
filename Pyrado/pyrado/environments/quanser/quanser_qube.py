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
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.quanser.base import RealEnv
from pyrado.policies.environment_specific import QQubePDCtrl, GoToLimCtrl
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn
from pyrado.utils.input_output import print_cbt


class QQubeReal(RealEnv, Serializable):
    """ Class for the real Quanser Qube a.k.a. Furuta pendulum """

    name: str = 'qq'

    def __init__(self,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None,
                 ip: str = '192.168.2.40'):
        """
        Constructor

        :param dt: sampling frequency on the Quanser device [Hz]
        :param max_steps: maximum number of steps executed on the device [-]
        :param task_args: arguments for the task construction
        :param ip: IP address of the Qube platform
        """
        Serializable._init(self, locals())

        # Initialize spaces, dt, max_step, and communication
        super().__init__(ip, rcv_dim=4, snd_dim=1, dt=dt, max_steps=max_steps, task_args=task_args)
        self._curr_act = np.zeros(self.act_space.shape)  # just for usage in render function
        self._sens_offset = np.zeros(4)  # last two entries are never calibrated but useful for broadcasting

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', np.array([0., np.pi, 0., 0.]))
        Q = task_args.get('Q', np.diag([3e-1, 1., 2e-2, 5e-3]))
        R = task_args.get('R', np.diag([4e-3]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([120./180*np.pi, 4*np.pi, 20*np.pi, 20*np.pi])  # [rad, rad, rad/s, rad/s]
        max_obs = np.array([1., 1., 1., 1., pyrado.inf, pyrado.inf])  # [-, -, -, -, rad/s, rad/s]
        max_act = np.array([4.])  # should be the same as QQubeSim [V]
        self._state_space = BoxSpace(-max_state, max_state,
                                     labels=[r'$\theta$', r'$\alpha$', r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._obs_space = BoxSpace(-max_obs, max_obs,
                                   labels=[r'$\sin\theta$', r'$\cos\theta$', r'$\sin\alpha$', r'$\cos\alpha$',
                                           r'$\dot{\theta}$', r'$\dot{\alpha}$'])
        self._act_space = BoxSpace(-max_act, max_act, labels=['$V$'])

    @property
    def task(self) -> Task:
        return self._task

    def observe(self, state):
        return np.array([np.sin(state[0]), np.cos(state[0]), np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    def reset(self, *args, **kwargs) -> np.ndarray:
        # Reset socket and task
        super().reset()

        # Run calibration routine to start in the center
        self.calibrate()

        # Start with a zero action and get the first sensor measurements
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Correct for offset
        meas -= self._sens_offset

        # Reset time counter
        self._curr_step = 0

        return self.observe(meas)

    def step(self, act: np.ndarray) -> tuple:
        info = dict(t=self._curr_step*self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act_lim = self.limit_act(act)
        self._curr_act = act_lim

        # Send actions and receive sensor measurements
        meas = self._qsoc.snd_rcv(act_lim)

        # Correct for offset
        meas -= self._sens_offset

        # Construct the state from the measurements
        self.state = meas
        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        # Add final reward if done
        if done:
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def calibrate(self):
        """ Calibration routine to move to the init position and determine the sensor offset """
        # Reset calibration
        self._sens_offset = np.zeros(4)  # last two entries are never calibrated but useful for broadcasting

        # Wait until Qube is at rest
        cnt = 0
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))
        while cnt < 1/self._dt:
            if np.abs(meas[3]) < 1e-8 and np.abs(meas[2]) < 1e-8:
                cnt += 1
            else:
                cnt = 0
            meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Record alpha offset (e.g. alpha == k * 2pi)
        self._sens_offset[1] = meas[1]

        # Create parts of the calibration controller
        go_right = GoToLimCtrl(positive=True, cnt_done=int(.5/self._dt))
        go_left = GoToLimCtrl(positive=False, cnt_done=int(.5/self._dt))
        go_center = QQubePDCtrl(self.spec, calibration_mode=True)

        # Go to both limits for theta calibration
        while not go_right.done:
            meas = self._qsoc.snd_rcv(
                go_right(to.from_numpy(meas - self._sens_offset)))  # already correct for alpha offset
        while not go_left.done:
            meas = self._qsoc.snd_rcv(
                go_left(to.from_numpy(meas - self._sens_offset)))  # already correct for alpha offset
        self._sens_offset[0] = (go_right.th_lim + go_left.th_lim)/2
        print_cbt(f'Sensor offset: {self._sens_offset}', 'g')

        while not go_center.done:
            meas = self._qsoc.snd_rcv(
                go_center(to.from_numpy(meas - self._sens_offset)))  # already correct for alpha offset
        print_cbt('Calibration done.', 'g')
