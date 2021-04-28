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

import os.path as osp
from typing import Optional

import numpy as np
import rcsenv
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ScaledExpQuadrErrRewFcn


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BallOnPlate"))


class BallOnPlateSim(RcsSim, Serializable):
    """Base class for the ball-on-plate environments simulated in Rcs using the Vortex or Bullet physics engine"""

    def __init__(self, task_args: dict, init_ball_vel: np.ndarray = None, max_dist_force: float = None, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param init_ball_vel: initial ball velocity applied to ball on `reset()`
        :param max_dist_force: maximum disturbance force, set to `None` (default) for no disturbance
        :param kwargs: keyword arguments forwarded to the `BallOnPlateSim` constructor
        """
        if init_ball_vel is not None:
            if not init_ball_vel.size() == 2:
                raise pyrado.ShapeErr(given=init_ball_vel, expected_match=np.empty(2))

        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType="BallOnPlate",
            graphFileName="gBotKuka.xml",
            physicsConfigFile="pBallOnPlate.xml",
            **kwargs,
        )

        # Initial ball velocity
        self._init_ball_vel = init_ball_vel

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        raise NotImplementedError

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            ball_mass=0.2,
            ball_radius=0.05,
            ball_com_x=0.0,
            ball_com_y=0.0,
            ball_com_z=0.0,
            ball_friction_coefficient=0.3,
            ball_rolling_friction_coefficient=0.05,
            ball_slip=50.0,
            ball_linearvelocitydamping=0.0,
            ball_angularvelocitydamping=0.0,
        )

    def _adapt_domain_param(self, params: dict) -> dict:
        if "ball_rolling_friction_coefficient" in params:
            br = params.get("ball_radius", None)
            if br is None:
                # If not set, get from the current simulation parameters
                br = self._sim.domainParam["ball_radius"]
            return dict(params, ball_rolling_friction_coefficient=params["ball_rolling_friction_coefficient"] * br)

        return params

    def _unadapt_domain_param(self, params: dict) -> dict:
        if "ball_rolling_friction_coefficient" in params and "ball_radius" in params:
            return dict(
                params,
                ball_rolling_friction_coefficient=params["ball_rolling_friction_coefficient"] / params["ball_radius"],
            )

        return params

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Call the parent class
        obs = RcsSim.reset(self, init_state, domain_param)

        # Apply a initial ball velocity if given
        if self._init_ball_vel is not None:
            self._sim.applyBallVelocity(self._init_ball_vel)
            # We could try to adapt obs here, but it's not really necessary

        return obs

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        # Sample angle and force uniformly
        angle = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(0, self._max_dist_force)
        return np.array([force * np.sin(angle), force * np.cos(angle), 0])


class BallOnPlate2DSim(BallOnPlateSim, Serializable):
    """Ball-on-plate environment with 2-dim actions"""

    name: str = "bop2d"

    def __init__(self, task_args: Optional[dict] = None, init_ball_vel: np.ndarray = None, **kwargs):
        """
        Constructor

        :param task_args: arguments for the task construction
        :param init_ball_vel: initial ball velocity applied to ball on `reset()`
        :param kwargs: keyword arguments forwarded to the `BallOnPlateSim` constructor
        """
        Serializable._init(self, locals())

        # Forward to the BallOnPlateSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict() if task_args is None else task_args,
            initBallVel=init_ball_vel,
            actionModelType="plate_angacc",
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.zeros(self.obs_space.flat_dim))

        Q = np.diag([1e-1, 1e-1, 1e1, 1e1, 0, 1e-3, 1e-3, 1e-2, 1e-2, 0])  # Pa, Pb, Bx, By, Bz, Pad, Pbd, Bxd, Byd, Bzd
        R = np.diag([1e-3, 1e-3])  # Padd, Pbdd

        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )


class BallOnPlate5DSim(BallOnPlateSim, Serializable):
    """Ball-on-plate environment with 5-dim actions"""

    name: str = "bop5d"

    def __init__(self, task_args: Optional[dict] = None, init_ball_vel: np.ndarray = None, **kwargs):
        """
        Constructor

        :param task_args: arguments for the task construction
        :param init_ball_vel: initial ball velocity applied to ball on `reset()`
        :param kwargs: keyword arguments forwarded to the `RcsSim` constructor
        """
        Serializable._init(self, locals())

        # Forward to the BallOnPlateSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict() if task_args is None else task_args,
            initBallVel=init_ball_vel,
            actionModelType="plate_acc5d",
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.zeros(self.obs_space.flat_dim))

        Q = np.diag(
            [1e-0, 1e-0, 1e-0, 1e-0, 1e-0, 1e3, 1e3, 1e3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-0, 1e-0, 1e-0]
        )  # Px, Py, Pz, Pa, Pb, Bx, By, Bz, Pxd, Pyd, Pzd, Pad, Pbd, Bxd, Byd, Bzd
        R = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3])  # Pxdd, Pydd, Pzdd, Padd, Pbdd
        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )
