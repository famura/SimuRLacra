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
Train an agent to solve the Qube swing-up task using Bayesian Domain Randomization.
"""
import numpy as np

import pyrado
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.domain_randomization.default_randomizers import (
    create_default_domain_param_map_qq,
    create_zero_var_randomizer,
)
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.spaces import BoxSpace
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name,
        f"{BayRn.name}-{PoWER.name}_{QQubeSwingUpAndBalanceCtrl.name}",
        f"sim2sim_seed-{args.seed}",
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 100.0, max_steps=600)
    env_sim = QQubeSwingUpSim(**env_sim_hparams)
    env_sim = DomainRandWrapperLive(env_sim, create_zero_var_randomizer(env_sim))
    dp_map = create_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real = QQubeSwingUpSim(**env_sim_hparams)
    env_real.domain_param = dict(
        mass_pend_pole=0.024 * 1.15,
        mass_rot_pole=0.095 * 1.15,
    )
    env_real_hparams = env_sim_hparams
    env_real = wrap_like_other_env(env_real, env_sim)

    # PoWER and energy-based controller setup
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827, acc_max=10.0)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **policy_hparam)
    subrtn_hparam = dict(
        max_iter=8,
        pop_size=100,
        num_init_states_per_domain=4,
        num_domains=16,
        num_is_samples=10,
        expl_std_init=2.0,
        expl_std_min=0.02,
        symm_sampling=False,
        num_workers=20,
    )
    subrtn = PoWER(ex_dir, env_sim, policy, **subrtn_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
    ddp_space = BoxSpace(
        bound_lo=np.array([0.7 * dp_nom["mass_pend_pole"], 1e-8, 0.7 * dp_nom["mass_rot_pole"], 1e-8]),
        bound_up=np.array([1.3 * dp_nom["mass_pend_pole"], 1e-7, 1.3 * dp_nom["mass_rot_pole"], 1e-7]),
    )

    # Algorithm
    bayrn_hparam = dict(
        max_iter=15,
        acq_fc="EI",
        acq_param=dict(beta=0.2),
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=2,
        warmstart=False,
        num_eval_rollouts_real=100,
        thold_succ_subrtn=-1,
    )

    # Save the environments and the hyper-parameters (do it before the init routine of BayRn)
    save_dicts_to_yaml(
        dict(env_sim=env_sim_hparams, env_real=env_real_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(subrtn=subrtn_hparam, subrtn_name=PoWER.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map),
        save_dir=ex_dir,
    )

    algo = BayRn(ex_dir, env_sim, env_real, subrtn, ddp_space, **bayrn_hparam)

    # Jeeeha
    algo.train(snapshot_mode="latest", seed=args.seed)
