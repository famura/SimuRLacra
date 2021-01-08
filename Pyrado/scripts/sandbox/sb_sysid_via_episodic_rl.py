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
Script to test the algorithm in a sim-to-sim toy task
"""
import torch as to
from pyrado.utils.argparser import get_argparser

import pyrado
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.episodic.nes import NES
from pyrado.algorithms.episodic.reps import REPS
from pyrado.algorithms.episodic.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper, DomainRandWrapperLive
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.special.domain_distribution import DomainDistrParamPolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.input_output import print_cbt


def create_bob_setup():
    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=500)
    env_real = BallOnBeamSim(**env_hparams)
    env_real.domain_param = dict(
        # l_beam=1.95,
        # ang_offset=-0.03,
        g=10.81
    )

    env_sim = BallOnBeamSim(**env_hparams)
    randomizer = DomainRandomizer(
        # NormalDomainParam(name="l_beam", mean=0, std=1e-6, clip_lo=1.5, clip_up=3.5),
        # UniformDomainParam(name="ang_offset", mean=0, halfspan=1e-6),
        NormalDomainParam(name="g", mean=0, std=1e-6),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {
        # 0: ("l_beam", "mean"), 1: ("l_beam", "std"),
        # 2: ("ang_offset", "mean"), 3: ("ang_offset", "halfspan")
        0: ("g", "mean"),
        1: ("g", "std"),
    }
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Policies (the behavioral policy needs to be deterministic)
    behavior_policy = LinearPolicy(env_sim.spec, feats=FeatureStack([identity_feat, sin_feat]))
    behavior_policy.param_values = to.tensor([3.8090, -3.8036, -1.0786, -2.4510, -0.9875, -1.3252, 3.1503, 1.4443])
    prior = DomainRandomizer(
        # NormalDomainParam(name="l_beam", mean=2.05, std=2.05/10),
        # UniformDomainParam(name="ang_offset", mean=0.03, halfspan=0.03/10),
        NormalDomainParam(name="g", mean=8.81, std=8.81 / 10),
    )
    # trafo_mask = [False, True, False, True]
    trafo_mask = [True, True]
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, trafo_mask=trafo_mask, prior=prior, scale_params=True)

    return env_sim, env_real, env_hparams, dp_map, behavior_policy, ddp_policy


def create_qqsu_setup():
    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=600)
    env_real = QQubeSwingUpSim(**env_hparams)
    env_real.domain_param = dict(
        Mr=0.095 * 0.9,  # 0.095*0.9 = 0.0855
        Mp=0.024 * 1.1,  # 0.024*1.1 = 0.0264
        Lr=0.085 * 0.9,  # 0.085*0.9 = 0.0765
        Lp=0.129 * 1.1,  # 0.129*1.1 = 0.1419
    )

    env_sim = QQubeSwingUpSim(**env_hparams)
    randomizer = DomainRandomizer(
        NormalDomainParam(name="Mr", mean=0.0, std=1e-9, clip_lo=1e-3),
        NormalDomainParam(name="Mp", mean=0.0, std=1e-9, clip_lo=1e-3),
        NormalDomainParam(name="Lr", mean=0.0, std=1e-9, clip_lo=1e-3),
        NormalDomainParam(name="Lp", mean=0.0, std=1e-9, clip_lo=1e-3),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {
        0: ("Mr", "mean"),
        1: ("Mr", "std"),
        2: ("Mp", "mean"),
        3: ("Mp", "std"),
        4: ("Lr", "mean"),
        5: ("Lr", "std"),
        6: ("Lp", "mean"),
        7: ("Lp", "std"),
    }
    # trafo_mask = [False, True, False, True, False, True, False, True]
    trafo_mask = [True] * 8
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Policies (the behavioral policy needs to be deterministic)
    behavior_policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)
    prior = DomainRandomizer(
        NormalDomainParam(name="Mr", mean=0.095, std=0.095 / 10),
        NormalDomainParam(name="Mp", mean=0.024, std=0.024 / 10),
        NormalDomainParam(name="Lr", mean=0.085, std=0.085 / 10),
        NormalDomainParam(name="Lp", mean=0.129, std=0.129 / 10),
    )
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, trafo_mask=trafo_mask, prior=prior, scale_params=False)

    return env_sim, env_real, env_hparams, dp_map, behavior_policy, ddp_policy


def create_cem_subrtn(ex_dir: str, env_sim: MetaDomainRandWrapper, ddp_policy: DomainDistrParamPolicy) -> [CEM, dict]:
    # Subroutine
    subrtn_hparam = dict(
        max_iter=20,
        pop_size=200,
        num_rollouts=1,
        num_is_samples=10,
        expl_std_init=5e-1,
        expl_std_min=1e-4,
        # extra_expl_std_init=0.1,
        # extra_expl_decay_iter=5,
        num_workers=8,
    )
    return CEM(ex_dir, env_sim, ddp_policy, **subrtn_hparam), subrtn_hparam


def create_reps_subrtn(ex_dir: str, env_sim: MetaDomainRandWrapper, ddp_policy: DomainDistrParamPolicy) -> [REPS, dict]:
    # Subroutine
    subrtn_hparam = dict(
        max_iter=20,
        eps=1.0,
        pop_size=500,
        num_rollouts=1,
        expl_std_init=5e-2,
        expl_std_min=1e-4,
        num_epoch_dual=1000,
        optim_mode="torch",
        lr_dual=5e-4,
        use_map=True,
        num_workers=8,
    )
    return REPS(ex_dir, env_sim, ddp_policy, **subrtn_hparam), subrtn_hparam


def create_nes_subrtn(ex_dir: str, env_sim: MetaDomainRandWrapper, ddp_policy: DomainDistrParamPolicy) -> [NES, dict]:
    # Subroutine
    subrtn_hparam = dict(
        max_iter=100,
        pop_size=None,
        num_rollouts=1,
        expl_std_init=2e-2,
        expl_std_min=1e-4,
        num_workers=8,
    )
    return NES(ex_dir, env_sim, ddp_policy, **subrtn_hparam), subrtn_hparam


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Choose an experiment
    # env_sim, env_real, env_hparams, dp_map, behavior_policy, ddp_policy = create_bob_setup()
    env_sim, env_real, env_hparams, dp_map, behavior_policy, ddp_policy = create_qqsu_setup()

    if args.mode == CEM.name:
        ex_dir = setup_experiment(env_real.name, f"{SysIdViaEpisodicRL.name}-{CEM.name}")
        subrtn, subrtn_hparam = create_cem_subrtn(ex_dir, env_sim, ddp_policy)
    elif args.mode == REPS.name:
        ex_dir = setup_experiment(env_real.name, f"{SysIdViaEpisodicRL.name}-{REPS.name}")
        subrtn, subrtn_hparam = create_reps_subrtn(ex_dir, env_sim, ddp_policy)
    elif args.mode == NES.name:
        ex_dir = setup_experiment(env_real.name, f"{SysIdViaEpisodicRL.name}-{NES.name}")
        subrtn, subrtn_hparam = create_nes_subrtn(ex_dir, env_sim, ddp_policy)
    else:
        raise NotImplementedError("Select mode cem, reps, or nes via the command line argument -m")

    # Set the seed
    pyrado.set_seed(1001, verbose=True)

    # Set the hyper-parameters of SysIdViaEpisodicRL
    num_eval_rollouts = 5
    algo_hparam = dict(
        metric=None,
        std_obs_filt=5,
        obs_dim_weight=[1, 1, 1, 1, 10, 10],
        num_rollouts_per_distr=len(dp_map) * 10,  # former 50
        num_workers=subrtn_hparam["num_workers"],
    )

    # Save the environments and the hyper-parameters
    save_list_of_dicts_to_yaml(
        [
            dict(env=env_hparams),
            dict(subrtn=subrtn_hparam, subrtn_name=subrtn.name),
            dict(algo=algo_hparam, algo_name=SysIdViaEpisodicRL.name, dp_map=dp_map),
        ],
        ex_dir,
    )

    algo = SysIdViaEpisodicRL(subrtn, behavior_policy, **algo_hparam)

    # Jeeeha
    while algo.curr_iter < algo.max_iter and not algo.stopping_criterion_met():
        algo.logger.add_value(algo.iteration_key, algo.curr_iter)

        # Create fake real-world data
        ro_real = []
        for _ in range(num_eval_rollouts):
            ro_real.append(rollout(env_real, behavior_policy, eval=True))

        algo.step(snapshot_mode="latest", meta_info=dict(rollouts_real=ro_real))
        algo.logger.record_step()
        algo._curr_iter += 1

    if algo.stopping_criterion_met():
        stopping_reason = "Stopping criterion met!"
    else:
        stopping_reason = "Maximum number of iterations reached!"

    if algo.policy is not None:
        print_cbt(
            f"{SysIdViaEpisodicRL.name} finished training a {ddp_policy.name} "
            f"with {ddp_policy.num_param} parameters. {stopping_reason}",
            "g",
        )
    else:
        print_cbt(f"{subrtn.name} finished training. {stopping_reason}", "g")
