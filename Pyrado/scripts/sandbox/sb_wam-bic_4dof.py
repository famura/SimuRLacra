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
Test Linear Policy with RBF Features for the WAM Ball-in-a-cup task with 4 dof.
"""
import numpy as np

import pyrado
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.policies.environment_specific import DualRBFLinearPolicy
from pyrado.utils.data_types import RenderMode
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Fix seed for reproducibility
    pyrado.set_seed(101)

    # Environment
    env = WAMBallInCupSim(
        num_dof=4,
        max_steps=3000,
        # Note, when tuning the task args: the `R` matrices are now 4x4 for the 4 dof WAM
        task_args=dict(
            R=np.zeros((4, 4)),
            R_dev=np.diag([0.2, 0.2, 1e-2, 1e-2])
        ),
    )

    # Stabilize ball and print out the stable state
    env.reset()
    act = np.zeros(env.spec.act_space.flat_dim)
    for i in range(1500):
        env.step(act)
        env.render(mode=RenderMode(video=True))

    # Printing out actual positions for 4-dof (..just needed to setup the hard-coded values in the class)
    print('Ball pos:', env.sim.data.get_body_xpos('ball'))
    print('Cup goal:', env.sim.data.get_site_xpos('cup_goal'))
    print('Joint pos (incl. first rope angle):', env.sim.data.qpos[:5])

    # Apply DualRBFLinearPolicy and plot the joint states over the desired ones
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.]), np.array([1.])))
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam, dim_mask=2)
    done, param = False, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(video=True), eval=True, reset_kwargs=dict(domain_param=param))
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, _, param = after_rollout_query(env, policy, ro)
