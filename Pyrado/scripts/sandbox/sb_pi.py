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
Script to test the Planar-Insert environment with the task activation action model
"""
import math

import rcsenv

import pyrado
from pyrado.environments.rcspysim.planar_insert import PlanarInsertIKActivationSim, PlanarInsertTASim
from pyrado.plotting.rollout_based import draw_potentials
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


rcsenv.setLogLevel(4)


def ik_activation_variant(dt, max_steps, max_dist_force, physics_engine, graph_file_name):
    # Set up environment
    env = PlanarInsertIKActivationSim(
        physicsEngine=physics_engine,
        graphFileName=graph_file_name,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeDynamicalSystemGoalDistance=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
    )
    env.reset(domain_param=dict(effector_friction=1.))

    # Set up policy
    def policy_fcn(t: float):
        return [0.1*dt, -0.01*dt, 3/180.*math.pi*math.sin(2.*math.pi*2.*t)]  # [m/s, m/s, rad/s]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate and plot potentials
    print(env.obs_space.labels)
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)


def ds_activation_variant(dt, max_steps, max_dist_force, physics_engine, graph_file_name):
    # Set up environment
    env = PlanarInsertTASim(
        physicsEngine=physics_engine,
        graphFileName=graph_file_name,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force,
        taskCombinationMethod='sum',  # 'sum', 'mean',  'product', or 'softmax'
        checkJointLimits=False,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeDynamicalSystemGoalDistance=False,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
    )
    env.reset(domain_param=dict(effector_friction=1.))

    # Set up policy
    def policy_fcn(t: float):
        return [0.7, 1, 0, 0.1, 0.5, 0.5]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate and plot potentials
    print(env.obs_space.labels)
    return rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=False)


if __name__ == '__main__':
    # Choose setup
    setup_type = 'ds_activation'  # ik_activation, or activation
    common_hparam = dict(
        dt=0.01,
        max_steps=1200,
        max_dist_force=None,
        physics_engine='Bullet',  # Bullet or Vortex
        graph_file_name='gPlanarInsert6Link.xml',  # gPlanarInsert6Link.xml or gPlanarInsert5Link.xml
    )

    if setup_type == 'ik_activation':
        ro = ik_activation_variant(**common_hparam)
    elif setup_type == 'ds_activation':
        ro = ds_activation_variant(**common_hparam)
        draw_potentials(ro)
    else:
        raise pyrado.ValueErr(given_name=setup_type, eq_constraint='ik_activation or ds_activation')
