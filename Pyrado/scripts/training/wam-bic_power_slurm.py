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
Train an agent to solve the WAM Ball-in-cup environment using Policy learning by Weighting Exploration with the Returns.
Set the seed using --seed ${SLURM_ARRAY_TASK_ID} to make use of SLURM array.
"""
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.domain_randomization.domain_parameter import UniformDomainParam, NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml, load_dict_from_yaml
from pyrado.policies.environment_specific import DualRBFLinearPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Directory of reference (BayRn) experiment
    ref_ex_name = '2020-09-16_13-46-57--rand-rl-rd-bm-js-jd'
    ref_ex_dir = osp.join(pyrado.EXP_DIR, WAMBallInCupSim.name, f'{BayRn.name}-{PoWER.name}', ref_ex_name)

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        WAMBallInCupSim.name,
        PoWER.name + '_' + DualRBFLinearPolicy.name,
        f'ref_{ref_ex_name}_argmax_seed-{args.seed}',
    )

    # Hyperparameters of reference experiment
    hparams = load_dict_from_yaml(osp.join(ref_ex_dir, 'hyperparams.yaml'))

    # Environment
    env_hparams = hparams['env']
    env = WAMBallInCupSim(**env_hparams)

    # Randomizer
    dp_nom = WAMBallInCupSim.get_nominal_domain_param()
    randomizer = DomainRandomizer(
        # UniformDomainParam(name='cup_scale', mean=0.95, halfspan=0.05),
        # UniformDomainParam(name='ball_mass', mean=2.1000e-02, halfspan=3.1500e-03, clip_lo=0),
        # UniformDomainParam(name='rope_length', mean=3.0000e-01, halfspan=1.5000e-02, clip_lo=0.27, clip_up=0.33),
        # UniformDomainParam(name='rope_damping', mean=1.0000e-04, halfspan=1.0000e-04, clip_lo=1e-2),
        # UniformDomainParam(name='joint_damping', mean=5.0000e-02, halfspan=5.0000e-02, clip_lo=1e-6),
        # UniformDomainParam(name='joint_stiction', mean=2.0000e-01, halfspan=2.0000e-01, clip_lo=0),
        #
        NormalDomainParam(name='rope_length', mean=2.9941e-01, std=1.0823e-02, clip_lo=0.27, clip_up=0.33),
        UniformDomainParam(name='rope_damping', mean=3.0182e-05, halfspan=4.5575e-05, clip_lo=0.),
        NormalDomainParam(name='ball_mass', mean=1.8412e-02, std=1.9426e-03, clip_lo=1e-2),
        UniformDomainParam(name='joint_stiction', mean=1.9226e-01, halfspan=2.5739e-02, clip_lo=0),
        UniformDomainParam(name='joint_damping', mean=9.4057e-03, halfspan=5.0000e-04, clip_lo=1e-6),
    )
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = hparams['policy']
    policy_hparam['rbf_hparam'].update({'scale': None})
    policy = DualRBFLinearPolicy(env.spec, **policy_hparam)
    policy.param_values = to.tensor(hparams['algo']['policy_param_init'])

    # Algorithm
    algo_hparam = hparams['subroutine']
    algo_hparam.update({'num_workers': 8})  # should be equivalent to the number of cores per job
    algo = PoWER(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed, snapshot_mode='latest')
