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
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters
"""
import torch as to

from pyrado.algorithms.power import PoWER
from pyrado.domain_randomization.default_randomizers import get_zero_var_randomizer, get_default_domain_param_map_qq
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.policies.features import FeatureStack, identity_feat, sign_feat, abs_feat, squared_feat, qubic_feat, \
    bell_feat, MultFeat
from pyrado.policies.linear import LinearPolicy
from pyrado.utils.experiments import wrap_like_other_env


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name,
                              f'{BayRn.name}-{PoWER.name}_{QQubeSwingUpAndBalanceCtrl.name}_sim2sim',
                              '100Hz_rand-Mp+Mr+', seed=1111)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=600)
    env_sim = QQubeSwingUpSim(**env_hparams)
    env_sim = DomainRandWrapperLive(env_sim, get_zero_var_randomizer(env_sim))
    dp_map = get_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = QQubeSwingUpSim(**env_hparams)
    env_real.domain_param = dict(Mp=0.026, Mr=0.097)
    # env_real = QQubeReal(**env_hparams)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy
    # policy_hparam = dict(
    #     feats=FeatureStack([identity_feat, sign_feat, abs_feat, squared_feat, qubic_feat,
    #                         MultFeat([2, 5]), MultFeat([3, 5]), MultFeat([4, 5])])
    # )
    # policy = LinearPolicy(spec=env_sim.spec, **policy_hparam)
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827, acc_max=10.)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **policy_hparam)

    # Subroutine
    subrtn_hparam = dict(
        max_iter=20,
        pop_size=50,
        num_rollouts=10,
        num_is_samples=25,
        expl_std_init=2.0,
        expl_std_min=0.02,
        symm_sampling=False,
        num_workers=8,
    )
    power = PoWER(ex_dir, env_sim, policy, **subrtn_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
    # bounds = to.tensor(
    #     [[0.8*dp_nom['Mp'], dp_nom['Mp']/2000],
    #      [1.2*dp_nom['Mp'], dp_nom['Mp']/1000]])
    bounds = to.tensor(
        [[0.9*dp_nom['Mp'], dp_nom['Mp']/5000, 0.9*dp_nom['Mr'], dp_nom['Mr']/5000],
         [1.1*dp_nom['Mp'], dp_nom['Mp']/4999, 1.1*dp_nom['Mr'], dp_nom['Mr']/4999]])
    # bounds = to.tensor(
    #     [[0.9*dp_nom['Mp'], dp_nom['Mp']/1000, 0.9*dp_nom['Mr'], dp_nom['Mr']/1000,
    #       0.9*dp_nom['Lp'], dp_nom['Lp']/1000, 0.9*dp_nom['Lr'], dp_nom['Lr']/1000],
    #      [1.1*dp_nom['Mp'], dp_nom['Mp']/20, 1.1*dp_nom['Mr'], dp_nom['Mr']/20,
    #       1.1*dp_nom['Lp'], dp_nom['Lp']/20, 1.1*dp_nom['Lr'], dp_nom['Lr']/20]])

    # Algorithm
    bayrn_hparam = dict(
        max_iter=15,
        acq_fc='UCB',
        acq_param=dict(beta=0.25),
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=2,
        warmstart=False,
        num_eval_rollouts_real=100 if isinstance(env_real, QQubeSwingUpSim) else 5,
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BDR)
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(subrtrn=subrtn_hparam, subrtrn_name=PoWER.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map)],
        ex_dir
    )

    algo = BayRn(ex_dir, env_sim, env_real, subrtn=power, bounds=bounds, **bayrn_hparam)

    # Jeeeha
    algo.train(snapshot_mode='best', seed=ex_dir.seed)
