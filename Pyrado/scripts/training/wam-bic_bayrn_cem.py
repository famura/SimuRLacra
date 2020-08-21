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
Train an agent to solve the WAM Ball-in-cup environment using Bayesian Domain Randomization.
"""
import numpy as np
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.cem import CEM
from pyrado.domain_randomization.default_randomizers import get_zero_var_randomizer, get_default_domain_param_map_wambic
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.barrett_wam.wam import WAMBallInCupReal
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    # ex_dir = setup_experiment(WAMBallInCupSim.name, f'{BayRn.name}_{CEM.name}', 'rand-rl-jd', seed=1001)
    ex_dir = setup_experiment(WAMBallInCupSim.name, f'{BayRn.name}_{CEM.name}_sim2sim', 'rand-rl-jd', seed=1001)

    # Environments
    env_hparams = dict(
        num_dof=4,
        max_steps=1500,
        fixed_init_state=False,
        task_args=dict(final_factor=0.05)
    )
    env_sim = WAMBallInCupSim(**env_hparams)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    # dp_map = get_default_domain_param_map_wambic()
    dp_map = {
        0: ('rope_length', 'mean'),
        1: ('rope_length', 'std'),
        2: ('joint_damping', 'mean'),
        3: ('joint_damping', 'halfspan'),
    }
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Set the boundaries for the GP (must be consistent with dp_map)
    dp_nom = WAMBallInCupSim.get_nominal_domain_param()
    bounds = to.tensor(
        # [[0.7*dp_nom['cup_scale'], dp_nom['cup_scale']/100, 0.8*dp_nom['rope_length'], dp_nom['rope_length']/100],
        #  [1.3*dp_nom['cup_scale'], dp_nom['cup_scale']/20, 1.2*dp_nom['rope_length'], dp_nom['rope_length']/10]]
        [[0.9*dp_nom['rope_length'], dp_nom['rope_length']/100, 0*dp_nom['joint_damping'], dp_nom['joint_damping']/100],
         [1.1*dp_nom['rope_length'], dp_nom['rope_length']/10, 4*dp_nom['joint_damping'], dp_nom['joint_damping']/10]]
    )

    # env_real = WAMBallInCupReal(ip=None)
    env_real = WAMBallInCupSim(**env_hparams)

    # Policy
    policy_hparam = dict(
        rbf_hparam=dict(num_feat_per_dim=9, bounds=(0., 1.), scale=None),
        dim_mask=2
    )
    policy = DualRBFLinearPolicy(env_sim.spec, **policy_hparam)
    # policy_init = to.load(osp.join(pyrado.EXP_DIR, WAMBallInCupSim.name, CEM.name,
    #                                'EXP_NAME', 'policy.pt'))

    # Subroutine
    subrtn_hparam = dict(
        max_iter=30,
        pop_size=100,
        num_rollouts=100,
        num_is_samples=10,
        expl_std_init=np.pi/6,
        expl_std_min=0.02,
        extra_expl_std_init=np.pi/6,
        extra_expl_decay_iter=15,
        full_cov=False,
        symm_sampling=False,
        num_workers=16,
    )
    cem = CEM(ex_dir, env_sim, policy, **subrtn_hparam)

    # Algorithm
    bayrn_hparam = dict(
        max_iter=15,
        acq_fc='EI',
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=5,
        warmstart=False,
        num_eval_rollouts_real=100 if isinstance(env_real, WAMBallInCupSim) else 5,
        num_eval_rollouts_sim=100,
        # policy_param_init=policy_init.param_values.data,
        subroutine_snapshot_mode='latest'
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BDR)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(subrtn=subrtn_hparam, subrtn_name=CEM.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, subrtn=cem, bounds=bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=ex_dir.seed)
