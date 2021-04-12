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
from init_args_serializer import Serializable
from typing import Optional

import pyrado
from pyrado.environments.barrett_wam import torque_space_wam_7dof, torque_space_wam_4dof
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.spaces.base import Space


class WAMSim(MujocoSimEnv, ABC, Serializable):
    """ Base class for WAM robotic arm from Barrett technologies"""

    def __init__(
        self,
        num_dof: int,
        model_path: str,
        frame_skip: int = 4,
        dt: Optional[float] = None,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
    ):
        """
        Constructor

        :param num_dof: number of degrees of freedom (4 or 7), depending on which Barrett WAM setup being used
        :param model_path: path to the MuJoCo xml model config file
        :param frame_skip: number of simulation frames for which the same action is held, results in a multiplier of
                           the time step size `dt`
        :param dt: by default the time step size is the one from the mujoco config file multiplied by the number of
                   frame skips (legacy from OpenAI environments). By passing an explicit `dt` value, this can be
                   overwritten. Possible use case if if you know that you recorded a trajectory with a specific `dt`.
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction
        """
        Serializable._init(self, locals())

        self._num_dof = num_dof
        self.camera_config = dict(
            trackbodyid=0,  # id of the body to track
            elevation=-30,  # camera rotation around the axis in the plane
            azimuth=-90,  # camera rotation around the camera's vertical axis
        )

        # Call MujocoSimEnv's constructor
        super().__init__(model_path, frame_skip, dt, max_steps, task_args)

    @property
    def num_dof(self) -> int:
        """ Get the number of degrees of freedom. """
        return self._num_dof

    @property
    def torque_space(self) -> Space:
        """ Get the space of joint torques. """
        return torque_space_wam_7dof if self._num_dof == 7 else torque_space_wam_4dof

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

    @classmethod
    def get_nominal_domain_param(cls, num_dof: int = 7) -> dict:
        if num_dof == 7:
            return dict(
                link_1_mass=10.76768767,  # [kg]
                link_2_mass=3.87493756,  # [kg]
                link_3_mass=1.80228141,  # [kg]
                link_4_mass=2.40016804,  # [kg]
                link_5_mass=0.12376019,  # [kg]
                link_6_mass=0.41797364,  # [kg]
                link_7_mass=0.06864753,  # [kg]
                joint_1_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_2_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_3_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_4_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_5_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_6_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_7_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_1_dryfriction=0.4,  # dry friction coefficient of motor joint 1 [-]
                joint_2_dryfriction=0.4,  # dry friction coefficient of motor joint 2 [-]
                joint_3_dryfriction=0.4,  # dry friction coefficient of motor joint 3 [-]
                joint_4_dryfriction=0.4,  # dry friction coefficient of motor joint 4 [-]
                joint_5_dryfriction=0.4,  # dry friction coefficient of motor joint 5 [-]
                joint_6_dryfriction=0.4,  # dry friction coefficient of motor joint 6 [-]
                joint_7_dryfriction=0.4,  # dry friction coefficient of motor joint 7 [-]
            )
        elif num_dof == 4:
            return dict(
                link_1_mass=10.76768767,  # [kg]
                link_2_mass=3.87493756,  # [kg]
                link_3_mass=1.80228141,  # [kg]
                link_4_mass=1.06513649,  # [kg]
                joint_1_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_2_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_3_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_4_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_1_dryfriction=0.4,  # dry friction coefficient of motor joint 1 [-]
                joint_2_dryfriction=0.4,  # dry friction coefficient of motor joint 2 [-]
                joint_3_dryfriction=0.4,  # dry friction coefficient of motor joint 3 [-]
                joint_4_dryfriction=0.4,  # dry friction coefficient of motor joint 4 [-]
            )
        else:
            raise pyrado.ValueErr(given=num_dof, eq_constraint="4 or 7")
