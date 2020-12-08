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
Simulate a policy for the WAM Ball-in-cup task.
Export the policy in form of desired joint position and velocities.
In addition, the actual joint position and velocities of the simulation are saved.
"""
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import pyrado
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the policy and the environment (for constructing the real-world counterpart)
    env, policy, _ = load_experiment(ex_dir, args)
    env = remove_all_dr_wrappers(env)
    env.domain_param = env.get_nominal_domain_param()
    print_cbt(f'Set up the environment with dt={env.dt} max_steps={env.max_steps}.', 'c')
    print_domain_params(env.domain_param)

    # Get the initial state from the command line, if given. Else, set None to delegate to the environment.
    if args.init_state is not None:
        init_state = env.init_space.sample_uniform()
        init_qpos = np.asarray(args.init_state)
        # The passed init state only concerns certain (non-zero) joint angles.
        # The last element is the angle of the first rope segment relative to the cup bottom plate
        if len(init_qpos) == 5:
            np.put(init_state, [1, 3, 5, 6, 7], init_qpos)  # 7 DoF
        elif len(init_qpos) == 3:
            np.put(init_state, [1, 3, 4], init_qpos)  # 4 DoF
        else:
            raise pyrado.ValueErr(given=args.init_state, given_name='init_state',
                                  msg='The passed init_state requires length 3 for 4dof and 5 for 7dof.')
    else:
        init_state = None

    # Fix seed for reproducibility
    pyrado.set_seed(args.seed)

    # Do the rollout
    ro = rollout(env, policy, eval=True, render_mode=RenderMode(video=args.animation),
                 reset_kwargs=dict(init_state=init_state))

    # Save the trajectories
    if not hasattr(ro, 'env_infos'):
        raise KeyError('Rollout does not have the field env_infos!')
    t = ro.env_infos['t']
    qpos, qvel = ro.env_infos['qpos'], ro.env_infos['qvel']
    qpos_des, qvel_des = ro.env_infos['qpos_des'], ro.env_infos['qvel_des']

    np.save(osp.join(ex_dir, 'qpos_sim.npy'), qpos)
    np.save(osp.join(ex_dir, 'qvel_sim.npy'), qvel)
    np.save(osp.join(ex_dir, 'qpos_des.npy'), qpos_des)
    np.save(osp.join(ex_dir, 'qvel_des.npy'), qvel_des)
