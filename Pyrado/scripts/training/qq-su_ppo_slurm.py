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
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml, load_dict_from_yaml
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Directory of reference (BayRn) experiment
    ref_ex_name = "2020-10-02_19-39-39--rand-Mp-Mr-Lp-Lr_lower-std"
    ref_ex_dir = osp.join(
        pyrado.EXP_DIR, QQubeSwingUpSim.name, f"{BayRn.name}-{PPO.name}_{FNNPolicy.name}", ref_ex_name
    )

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name,
        f"{PPO.name}_{FNNPolicy.name}",
        f"ref_{ref_ex_name}_argmax_seed-{args.seed}"
        # QQubeSwingUpSim.name, f'{UDR.name}-{PPO.name}_{FNNPolicy.name}', f'ref_{ref_ex_name}_seed-{args.seed}'
        # QQubeSwingUpSim.name, f'{PPO.name}_{FNNPolicy.name}', f'ref_{ref_ex_name}_nominal_seed-{args.seed}'
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Hyperparameters of reference experiment
    hparams = load_dict_from_yaml(osp.join(ref_ex_dir, "hyperparams.yaml"))

    # Environment
    env_hparams = hparams["env_sim"]
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Randomizer
    dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
    randomizer = DomainRandomizer(
        # UniformDomainParam(name='Mp', mean=0.024, halfspan=0.0048),
        # UniformDomainParam(name='Mr', mean=0.095, halfspan=0.0190),
        # UniformDomainParam(name='Lp', mean=0.129, halfspan=0.0258),
        # UniformDomainParam(name='Lr', mean=0.085, halfspan=0.0170),
        # #
        NormalDomainParam(name="Mp", mean=0.0227, std=0.0009),
        NormalDomainParam(name="Mr", mean=0.0899, std=0.0039),
        NormalDomainParam(name="Lp", mean=0.1474, std=0.0046),
        NormalDomainParam(name="Lr", mean=0.0777, std=0.003),
    )
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy = to.load(osp.join(ref_ex_dir, "policy.pt"))
    policy.init_param()

    # Critic
    vfcn = to.load(osp.join(ref_ex_dir, "valuefcn.pt"))
    vfcn.init_param()
    critic = GAE(vfcn, **hparams["critic"])

    # Algorithm
    algo_hparam = hparams["subrtn"]
    algo_hparam.update({"num_workers": 1})  # should be equivalent to the number of cores per job
    # algo_hparam.update({'max_iter': 300})
    # algo_hparam.update({'max_iter': 600})
    # algo_hparam.update({'min_steps': 3*algo_hparam['min_steps']})
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=hparams["policy"]),
        dict(critic=hparams["critic"]),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed, snapshot_mode="latest")
