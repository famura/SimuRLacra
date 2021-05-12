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
from typing import Sequence, Union

import numpy as np

import pyrado
from pyrado.spaces.empty import EmptySpace
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.endless_flipping import FlippingTask
from pyrado.tasks.final_reward import FinalRewMode, FinalRewTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import (
    AbsErrRewFcn,
    CosOfOneEleRewFcn,
    ExpQuadrErrRewFcn,
    QuadrErrRewFcn,
    RewFcn,
    ZeroPerStepRewFcn,
)
from pyrado.tasks.utils import never_succeeded, proximity_succeeded
from pyrado.utils.data_types import EnvSpec


def create_check_all_boundaries_task(env_spec: EnvSpec, penalty: float) -> FinalRewTask:
    """
    Create a task that is checking if any of the state space bounds is violated.
    This checks every limit and not just of a subspace of the state state as it could happen when using a `MaskedTask`.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param penalty: scalar cost (positive values) for violating the bounds
    :return: masked task that only considers a subspace of all observations
    """
    return FinalRewTask(
        DesStateTask(env_spec, np.zeros(env_spec.state_space.shape), ZeroPerStepRewFcn(), never_succeeded),
        FinalRewMode(always_negative=True),
        factor=penalty,
    )


def create_task_space_discrepancy_task(env_spec: EnvSpec, rew_fcn: RewFcn) -> MaskedTask:
    """
    Create a task which punishes the discrepancy between the actual and the commanded state of the observed body.
    The observed body is specified in in the associated experiment configuration file in RcsPySim.
    This task only looks at the X and Z coordinates.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param rew_fcn: reward function
    :return: masked task that only considers a subspace of all observations
    """
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    obs_labels = [obs_label for obs_label in env_spec.state_space.labels if "DiscrepTS" in obs_label]
    if not obs_labels:
        raise pyrado.ValueErr(msg="No state space labels found that contain 'DiscrepTS'")

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels)),
    )

    # Create an endlessly running desired state task (no task space discrepancy is desired)
    task = DesStateTask(spec, np.zeros(spec.state_space.shape), rew_fcn, never_succeeded)

    # Mask selected discrepancy observation
    return MaskedTask(env_spec, task, obs_labels)


def create_collision_task(env_spec: EnvSpec, factor: float) -> MaskedTask:
    """
    Create a task which punishes collision costs given a collision model with pairs of bodies.
    This task only looks at the instantaneous collision cost.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param factor: cost / reward function scaling factor
    :return: masked task that only considers a subspace of all observations
    """
    if not factor >= 0:
        raise pyrado.ValueErr(given=factor, ge_constraint="0")

    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    obs_labels = ["CollCost"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels)),
    )

    rew_fcn = AbsErrRewFcn(q=np.array([factor]), r=np.zeros(spec.act_space.shape))

    # Create an endlessly running desired state task (no collision is desired)
    task = DesStateTask(spec, np.zeros(spec.state_space.shape), rew_fcn, never_succeeded)

    # Mask selected collision cost observation
    return MaskedTask(env_spec, task, obs_labels)


def create_forcemin_task(env_spec: EnvSpec, obs_labels: Sequence[str], Q: np.ndarray) -> MaskedTask:
    """
    Create a task which punishes the amount of used force.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param obs_labels: labels for selection, e.g. ['WristLoadCellLBR_R_Fy']. This needs to match the observations'
                       names in RcsPySim
    :param Q: weight matrix of dim NxN with N=num_forces for the quadratic force costs
    :return: masked task that only considers a subspace of all observations
    """
    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels)),
    )

    # Create an endlessly running desired state task
    rew_fcn = QuadrErrRewFcn(Q=Q, R=np.zeros((spec.act_space.flat_dim, spec.act_space.flat_dim)))
    task = DesStateTask(spec, np.zeros(spec.state_space.shape), rew_fcn, never_succeeded)

    # Mask selected collision cost observation
    return MaskedTask(env_spec, task, obs_labels)


def create_lifting_task(
    env_spec: EnvSpec,
    obs_labels: Sequence[str],
    des_height: Union[float, np.ndarray],
    succ_thold: float = 0.01,
) -> MaskedTask:
    """
    Create a task for lifting an object.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param obs_labels: labels for selection, e.g. ['Box_Z']. This needs to match the observations' names in RcsPySim
    :param des_height: desired height of the object (depends of the coordinate system). If reached, the task is over.
    :param succ_thold: once the object of interest is closer than this threshold, the task is considered successfully
    :return: masked task that only considers a subspace of all observations
    """
    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels)),
    )

    # Create a desired state task
    state_des = np.asarray(des_height)
    Q = np.diag([6e2])
    R = 1e-1 * np.eye(spec.act_space.flat_dim)
    rew_fcn = ExpQuadrErrRewFcn(Q, R)
    task = DesStateTask(spec, state_des, rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Return the masked tasks
    return MaskedTask(env_spec, task, obs_labels)


def create_flipping_task(
    env_spec: EnvSpec,
    obs_labels: Sequence[str],
    des_angle_delta: float = np.pi / 2.0,
    endless: bool = True,
) -> MaskedTask:
    """
    Create a task for rotating an object.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param obs_labels: labels for selection, e.g. ['Box_A']. This needs to match the observations' names in RcsPySim
    :param des_angle_delta: desired angle to rotate. If reached, the task is reset, and rotating continues.
    :param endless: if `True`, the task will promote endlessly repeated flipping about the desired angle, else only one
                    flip is desired
    :return: masked task that only considers a subspace of all observations
    """
    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels)),
    )

    # Create a desired state task
    rew_fcn = CosOfOneEleRewFcn(idx=0)
    task = FlippingTask(spec, des_angle_delta, rew_fcn, endless=endless)

    # Return the masked tasks
    return MaskedTask(env_spec, task, obs_labels)


def create_home_pos_task(env_spec: EnvSpec, obs_labels: Sequence[str], state_des: np.ndarray) -> MaskedTask:
    """
    Create a task for moving the robot to safe position.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param obs_labels: labels for selection, e.g. ['PowerGrasp_R_Y', 'PowerGrasp_R_Z']. This needs to match the
                       observations' names in RcsPySim
    :param state_des: desired state (depends of the coordinate system). If reached, the task is over.
    :return: masked task that only considers a subspace of all observations
    """
    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels)),
    )

    # Create a desired state task
    Q = 1e1 * np.eye(len(state_des))
    R = 1e-1 * np.eye(spec.act_space.flat_dim)
    rew_fcn = ExpQuadrErrRewFcn(Q, R)
    task = DesStateTask(spec, state_des, rew_fcn)

    # Return the masked tasks
    return MaskedTask(env_spec, task, obs_labels)


def create_goal_dist_task(env_spec: EnvSpec, ds_label: int, rew_fcn: RewFcn, succ_thold: float = 0.01) -> MaskedTask:
    """
    Create a task that rewards minimizing the `GoalDistance` of dynamical system movements primitives (see RcsPySim).

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param ds_label: label of the dynamical system (see RcsPySim)
    :param rew_fcn: reward function
    :param succ_thold: once the object of interest is closer than this threshold, the task is considered successfully
    :return: masked task that only considers a subspace of all observations
    """
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    obs_labels = [f"GD_DS{ds_label}"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels))
        if env_spec.state_space is not EmptySpace
        else EmptySpace,
    )

    # Create a desired state task with the goal [0, 0]
    task = DesStateTask(spec, np.zeros(2), rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Mask selected goal distance
    return MaskedTask(env_spec, task, obs_labels)


def create_goal_dist_distvel_task(
    env_spec: EnvSpec, ds_label: int, rew_fcn: RewFcn, succ_thold: float = 0.01
) -> MaskedTask:
    """
    Create a task that rewards minimizing the `GoalDistance` as well as its rate of change of dynamical system
    movements primitives (see RcsPySim).

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param ds_label: label of the dynamical system (see RcsPySim)
    :param rew_fcn: reward function
    :param succ_thold: once the object of interest is closer than this threshold, the task is considered successfully
    :return: masked task that only considers a subspace of all observations
    """
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    obs_labels = [f"GD_DS{ds_label}", f"GD_DS{ds_label}d"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(obs_labels))
        if env_spec.state_space is not EmptySpace
        else EmptySpace,
    )

    # Create a desired state task with the goal [0, 0]
    task = DesStateTask(spec, np.zeros(2), rew_fcn, functools.partial(proximity_succeeded, thold_dist=succ_thold))

    # Mask selected goal distance velocities
    return MaskedTask(env_spec, task, obs_labels)
