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

import functools
import os.path as osp
from typing import Sequence

import numpy as np
import rcsenv
from init_args_serializer import Serializable

from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import (
    create_check_all_boundaries_task,
    create_collision_task,
    create_goal_dist_task,
    create_task_space_discrepancy_task,
)
from pyrado.tasks.reward_functions import AbsErrRewFcn, ExpQuadrErrRewFcn, MinusOnePerStepRewFcn
from pyrado.tasks.utils import proximity_succeeded
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)  # pylint: disable=no-member
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BoxShelving"))  # pylint: disable=no-member


def create_box_upper_shelve_task(env_spec: EnvSpec, continuous_rew_fcn: bool, succ_thold: float):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ["Box_X", "Box_Y", "Box_Z", "Box_A", "Box_B", "Box_C"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space, env_spec.act_space, env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    state_des = np.zeros(6)  # zeros since we observe the box position relative to the goal
    if continuous_rew_fcn:
        Q = np.diag([5e0, 5e0, 5e0, 1e-1, 1e-1, 1e-1])
        R = 5e-2 * np.eye(spec.act_space.flat_dim)
        rew_fcn = ExpQuadrErrRewFcn(Q, R)
    else:
        rew_fcn = MinusOnePerStepRewFcn
    dst = DesStateTask(spec, state_des, rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Return the masked tasks
    return MaskedTask(env_spec, dst, idcs)


class BoxShelvingSim(RcsSim, Serializable):
    """ Base class for 2-armed humanoid robot putting a box into a shelve """

    def __init__(
        self, task_args: dict, ref_frame: str, position_mps: bool, tasks_left: [Sequence[dict], None], **kwargs
    ):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the MPs, e.g. 'world', 'box', or 'upperGoal'
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = False,
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType="BoxShelving",
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BoxShelving"),
            hudColor="BLACK_RUBBER",
            refFrame=ref_frame,
            positionTasks=position_mps,
            tasksLeft=tasks_left,
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        continuous_rew_fcn = task_args.get("continuous_rew_fcn", True)
        task_box = create_box_upper_shelve_task(self.spec, continuous_rew_fcn, succ_thold=5e-2)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        task_collision = create_collision_task(self.spec, factor=1.0)
        task_ts_discrepancy = create_task_space_discrepancy_task(
            self.spec, AbsErrRewFcn(q=0.5 * np.ones(3), r=np.zeros(self.act_space.shape))
        )

        return ParallelTasks(
            [task_box, task_check_bounds, task_collision, task_ts_discrepancy], hold_rew_when_done=False
        )

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(box_length=0.32, box_width=0.2, box_height=0.1, box_mass=1.0, box_friction_coefficient=0.8)


class BoxShelvingPosDSSim(BoxShelvingSim, Serializable):
    """ Humanoid robot putting a box into a shelve using one arm and position-level movement primitives """

    name: str = "bs-pos"

    def __init__(
        self, ref_frame: str, tasks_left: [Sequence[dict], None] = None, continuous_rew_fcn: bool = True, **kwargs
    ):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'box', or 'upperGoal'
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = False,
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Get the nominal domain parameters for the task specification
        dp_nom = BoxShelvingSim.get_nominal_domain_param()

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        if tasks_left is None:
            if not ref_frame == "upperGoal":
                print_cbt(f"Using tasks specified in the upperGoal frame in the {ref_frame} frame!", "y", bright=True)
            tasks_left = [
                # Left power grasp position
                {
                    "function": "msd",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 100.0,
                    "goal": np.array([0.65, 0, 0.0]),  # far in front
                },
                {
                    "function": "msd",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 100.0,
                    "goal": np.array([0.35, 0, -0.15]),  # below and in front
                },
                {
                    "function": "msd",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 100.0,
                    "goal": np.array([0.2, 0, 0.1]),  # close and slightly above
                },
                # Left power grasp orientation
                {
                    "function": "msd",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 100.0,
                    "goal": np.pi / 180 * np.array([-90, 0, -90.0]),  # upright
                },
                {
                    "function": "msd",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 100.0,
                    "goal": np.pi / 180 * np.array([-90, 0, -160.0]),  # tilted forward (into shelve)
                },
                # Joints SDH
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 50.0,
                    "mass": 1.0,
                    "damping": 50.0,
                    "goal": 10 / 180 * np.pi * np.array([0, 2, -1.5, 2, 0, 2, 0]),
                },
            ]

        # Forward to the BoxShelvingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn, tasks_left=tasks_left),
            tasks_left=tasks_left,
            ref_frame=ref_frame,
            position_mps=True,
            **kwargs,
        )


class BoxShelvingVelDSSim(BoxShelvingSim, Serializable):
    """ Humanoid robot putting a box into a shelve using one arm and velocity-level movement primitives """

    name: str = "bs-vel"

    def __init__(
        self,
        ref_frame: str,
        bidirectional_mps: bool,
        tasks_left: [Sequence[dict], None] = None,
        continuous_rew_fcn: bool = True,
        **kwargs,
    ):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'box', or 'upperGoal'
        :param bidirectional_mps: if `True` then the MPs can be activated "forward" and "backward", thus the `ADN`
                                  output activations must be in [-1, 1] and the output nonlinearity should be a tanh.
                                  If `false` then the behavior is the same as for position-level MPs.
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = False,
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Get the nominal domain parameters for the task specification
        dp_nom = BoxShelvingSim.get_nominal_domain_param()

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        if tasks_left is None:
            dt = kwargs.get("dt", 0.01)  # 100 Hz is the default

            if bidirectional_mps:
                tasks_left = [
                    # Xd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                    # Yd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                    # Zd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                    # Ad
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([10 / 180 * np.pi])},  # [rad/s]
                    # Bd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([10 / 180 * np.pi])},  # [rad/s]
                    # Joints SDH
                    {
                        "function": "msd_nlin",
                        "attractorStiffness": 50.0,
                        "mass": 1.0,
                        "damping": 50.0,
                        "goal": 10 / 180 * np.pi * np.array([0, 2, -1.5, 2, 0, 2, 0]),
                    },
                ]
            else:
                tasks_left = [
                    # Xd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([-0.15])},  # [m/s]
                    # Yd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([-0.15])},  # [m/s]
                    # Zd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([-0.15])},  # [m/s]
                    # Ad
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([10 / 180 * np.pi])},  # [rad/s]
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([-10 / 180 * np.pi])},  # [rad/s]
                    # Bd
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([10 / 180 * np.pi])},  # [rad/s]
                    {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([-10 / 180 * np.pi])},  # [rad/s]
                    # Joints SDH
                    {
                        "function": "msd_nlin",
                        "attractorStiffness": 50.0,
                        "mass": 1.0,
                        "damping": 50.0,
                        "goal": 10 / 180 * np.pi * np.array([0, 2, -1.5, 2, 0, 2, 0]),
                    },
                ]

        # Forward to the BoxShelvingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            tasks_left=tasks_left,
            ref_frame=ref_frame,
            position_mps=False,
            bidirectionalMPs=bidirectional_mps,
            **kwargs,
        )
