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

"""
Script to test the Planar-3-Link environment with different action models
"""
import math
import numpy as np
import torch as to

import rcsenv
import pyrado
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_3_link import (
    Planar3LinkJointCtrlSim,
    Planar3LinkIKActivationSim,
    Planar3LinkTASim,
)
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.plotting.rollout_based import draw_potentials
from pyrado.policies.recurrent.adn import ADNPolicy, pd_cubic
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(4)


def create_joint_control_setup(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkJointCtrlSim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        checkJointLimits=True,
    )
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        return [
            10 / 180 * math.pi,
            10 / 180 * math.pi,  # same as init config
            10 / 180 * math.pi + 45.0 / 180.0 * math.pi * math.sin(2.0 * math.pi * 0.2 * t),
        ]  # oscillation in last link

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


def create_ik_activation_setup(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkIKActivationSim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        taskCombinationMethod="product",
        positionTasks=True,
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeTaskSpaceDiscrepancy=True,
    )
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        return [0.3 + 0.2 * math.sin(2.0 * math.pi * 0.2 * t), 0, 1]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


def create_ds_activation_setup(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkTASim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        positionTasks=True,
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeCollisionCost=True,
        observeTaskSpaceDiscrepancy=True,
        observeDynamicalSystemDiscrepancy=False,
    )
    print(env.obs_space.labels)

    # Set up policy
    def policy_fcn(t: float):
        if t < 3:
            return [0, 1, 0]
        elif t < 7:
            return [1, 0, 0]
        elif t < 10:
            return [0.5, 0.5, 0]
        else:
            return [0, 0, 1]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)


def create_manual_activation_setup(dt, max_steps, max_dist_force, physics_engine):
    # Set up environment
    env = Planar3LinkTASim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        positionTasks=True,
        observeTaskSpaceDiscrepancy=True,
    )
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        pot = np.fromstring(input("Enter potentials for next step: "), dtype=np.double, count=3, sep=" ")
        return 1 / (1 + np.exp(-pot))

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)


def create_adn_setup(dt, max_steps, max_dist_force, physics_engine, normalize_obs=True, obsnorm_cpp=True):
    pyrado.set_seed(0)

    # Explicit normalization bounds
    elb = {
        "EffectorLoadCell_Fx": -100.0,
        "EffectorLoadCell_Fz": -100.0,
        "Effector_Xd": -1,
        "Effector_Zd": -1,
        "GD_DS0d": -1,
        "GD_DS1d": -1,
        "GD_DS2d": -1,
    }
    eub = {
        "GD_DS0": 3.0,
        "GD_DS1": 3,
        "GD_DS2": 3,
        "EffectorLoadCell_Fx": 100.0,
        "EffectorLoadCell_Fz": 100.0,
        "Effector_Xd": 0.5,
        "Effector_Zd": 0.5,
        "GD_DS0d": 0.5,
        "GD_DS1d": 0.5,
        "GD_DS2d": 0.5,
        "PredCollCost_h50": 1000.0,
    }

    extra_kwargs = {}
    if normalize_obs and obsnorm_cpp:
        extra_kwargs["normalizeObservations"] = True
        extra_kwargs["obsNormOverrideLower"] = elb
        extra_kwargs["obsNormOverrideUpper"] = eub

    # Set up environment
    env = Planar3LinkTASim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        positionTasks=True,
        collisionAvoidanceIK=True,
        taskCombinationMethod="sum",
        observeTaskSpaceDiscrepancy=True,
        **extra_kwargs,
    )

    if normalize_obs and not obsnorm_cpp:
        env = ObsNormWrapper(env, explicit_lb=elb, explicit_ub=eub)

    # Set up random policy
    policy_hparam = dict(
        tau_init=10.0,
        activation_nonlin=to.sigmoid,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, **policy_hparam)
    print_cbt("Running ADNPolicy with random initialization", "c", bright=True)

    # Simulate and plot potentials
    ro = rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)
    draw_potentials(ro)

    return ro


if __name__ == "__main__":
    # Choose setup
    setup_type = "ik_activation"  # joint, ik_activation, activation, manual, adn
    common_hparam = dict(
        dt=0.01,
        max_steps=1800,
        max_dist_force=None,
        physics_engine="Bullet",  # Bullet or Vortex
    )

    if setup_type == "joint":
        ro = create_joint_control_setup(**common_hparam)
    elif setup_type == "ik_activation":
        ro = create_ik_activation_setup(**common_hparam)
    elif setup_type == "ds_activation":
        ro = create_ds_activation_setup(**common_hparam)
    elif setup_type == "manual":
        ro = create_manual_activation_setup(**common_hparam)
    elif setup_type == "adn":
        ro = create_adn_setup(**common_hparam, normalize_obs=True, obsnorm_cpp=False)
    else:
        raise pyrado.ValueErr(
            given=setup_type, eq_constraint="'joint', 'ik_activation', 'ds_activation', 'manual', 'adn'"
        )
