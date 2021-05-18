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
Train an agent to solve the Quanser Qube swing-up task using Soft-Actor-Critic.
"""
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.meta.sprl import SPRL
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.sac import SAC
from pyrado.domain_randomization.domain_parameter import SelfPacedDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.two_headed_fnn import TwoHeadedFNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.spaces.box import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument("--frequency", default=250, type=int)
    parser.set_defaults(max_steps=600)
    parser.add_argument("--sac_iterations", default=150, type=int)
    parser.add_argument("--sprl_iterations", default=50, type=int)
    parser.add_argument("--cov_only", action="store_true")
    args = parser.parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name,
        f"{SAC.name}_{FNNPolicy.name}",
        f"{args.frequency}Hz_{args.max_steps}ROLen_{args.sac_iterations}SACIter_{args.sprl_iterations}SPRLIter_cov_only{args.cov_only}_seed_{args.seed}",
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / float(args.frequency), max_steps=args.max_steps)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(shared_hidden_sizes=[128, 128], shared_hidden_nonlin=to.relu)  # FNN
    policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)

    qfnc_param = dict(hidden_sizes=[128, 128], hidden_nonlin=to.relu)
    combined_space = BoxSpace.cat([env.obs_space, env.act_space])
    q1 = FNNPolicy(spec=EnvSpec(combined_space, ValueFunctionSpace), **qfnc_param)
    q2 = FNNPolicy(spec=EnvSpec(combined_space, ValueFunctionSpace), **qfnc_param)

    # Subroutine
    algo_hparam = dict(
        max_iter=args.sac_iterations,
        memory_size=10_000,
        gamma=0.99,
        num_updates_per_step=None,  # uses internal heuristic
        tau=0.995,
        ent_coeff_init=0.2,
        learn_ent_coeff=False,
        target_update_intvl=1_000,
        num_init_memory_steps=10_000,
        standardize_rew=False,
        min_steps=30 * env.max_steps,
        batch_size=500,
        lr=6e-04,
        max_grad_norm=1.0,
        num_workers=8,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    env_sprl_params = [
        dict(
            name="g",
            target_mean=to.tensor([9.81]),
            target_cov_chol_flat=to.tensor([1.0]),
            init_mean=to.tensor([9.81]),
            init_cov_chol_flat=to.tensor([0.05]),
        )
    ]
    env = DomainRandWrapperLive(env, randomizer=DomainRandomizer(*[SelfPacedDomainParam(**p) for p in env_sprl_params]))

    sprl_hparam = dict(
        kl_constraints_ub=8000,
        performance_lower_bound=500,
        std_lower_bound=0.4,
        kl_threshold=200,
        max_iter=args.sprl_iterations,
        optimize_mean=not args.cov_only,
    )
    algo = SPRL(env, SAC(ex_dir, env, policy, q1, q2, **algo_hparam), **sprl_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(subrtn=algo_hparam, subrtn_name=SAC.name),
        dict(algo=sprl_hparam, algo_name=algo.name, env_sprl_params=env_sprl_params),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="latest", seed=args.seed)
