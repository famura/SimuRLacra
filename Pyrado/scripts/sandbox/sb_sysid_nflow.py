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
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import ReversePermutation, MaskedAffineAutoregressiveTransform, CompositeTransform

import pyrado
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.episodic.nes import NES
from pyrado.algorithms.episodic.sysid_via_nflows import StochSysIdViaNFlows
from pyrado.domain_randomization.domain_randomizer import DistributionFreeDomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.policies.feed_forward.nflow import NFlowPolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


def create_bob_setup():
    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=500)
    env_real = BallOnBeamSim(**env_hparams)
    env_real.domain_param = dict(
        g=10.81,
        ang_offset=-0.03,
    )
    env_sim = BallOnBeamSim(**env_hparams)

    # Policies (the behavioral policy needs to be deterministic)
    behavior_policy = LinearPolicy(env_sim.spec, feats=FeatureStack([identity_feat, sin_feat]))
    behavior_policy.param_values = to.tensor([3.8090, -3.8036, -1.0786, -2.4510, -0.9875, -1.3252, 3.1503, 1.4443])

    # Create the normalizing flow
    dp_map = {
        0: "g",
        1: "ang_offset",
    }
    # trafo_mask = [True, False]
    trafo_mask = [False, False]
    base_dist = StandardNormal(shape=[2])
    transforms = []
    for _ in range(5):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2, hidden_features=4))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    nflow_policy = NFlowPolicy(flow, mapping=dp_map, trafo_mask=trafo_mask)

    # Add the randomizer
    randomizer = DistributionFreeDomainRandomizer(
        mapping=dp_map,
        rand_engine=nflow_policy,
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)

    return env_sim, env_real, env_hparams, dp_map, behavior_policy, nflow_policy


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

    # Policies (the behavioral policy needs to be deterministic)
    behavior_policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)

    # Create the normalizing flow
    dp_map = {
        0: "Mr",
        1: "Mp",
        2: "Lr",
        3: "Lp",
    }
    # trafo_mask = [False, True, False, True, False, True, False, True]
    trafo_mask = [True] * 8
    base_dist = StandardNormal(shape=[2])
    transforms = []
    for _ in range(5):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2, hidden_features=4))
    transform = CompositeTransform(transforms)
    flow = Flow(transform, base_dist)
    nflow_policy = NFlowPolicy(flow, mapping=dp_map, trafo_mask=trafo_mask)

    # Add the randomizer
    randomizer = DistributionFreeDomainRandomizer(
        mapping=dp_map,
        rand_engine=nflow_policy,
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)

    return env_sim, env_real, env_hparams, dp_map, behavior_policy, nflow_policy


def create_cem_subrtn(ex_dir: str, env_sim: DomainRandWrapperLive, nflow_policy: NFlowPolicy) -> [CEM, dict]:
    # Subroutine
    subrtn_hparam = dict(
        max_iter=50,
        pop_size=200,
        num_rollouts=1,
        num_is_samples=10,
        expl_std_init=1e-2,
        expl_std_min=1e-4,
        # extra_expl_std_init=0.1,
        # extra_expl_decay_iter=5,
        num_workers=4,
    )
    return CEM(ex_dir, env_sim, nflow_policy, **subrtn_hparam), subrtn_hparam


def create_nes_subrtn(ex_dir: str, env_sim: DomainRandWrapperLive, nflow_policy: NFlowPolicy) -> [NES, dict]:
    # Subroutine
    subrtn_hparam = dict(
        max_iter=200,
        pop_size=None,
        num_rollouts=1,
        expl_std_init=1e-2,
        expl_std_min=1e-4,
        num_workers=4,
    )
    return NES(ex_dir, env_sim, nflow_policy, **subrtn_hparam), subrtn_hparam


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Choose an experiment
    env_sim, env_real, env_hparams, dp_map, behavior_policy, nflow_policy = create_bob_setup()
    # env_sim, env_real, env_hparams, dp_map, behavior_policy, nflow_policy = create_qqsu_setup()

    if args.mode == CEM.name:
        ex_dir = setup_experiment(env_real.name, f"{StochSysIdViaNFlows.name}-{CEM.name}")
        subrtn, subrtn_hparam = create_cem_subrtn(ex_dir, env_sim, nflow_policy)
    elif args.mode == NES.name:
        ex_dir = setup_experiment(env_real.name, f"{StochSysIdViaNFlows.name}-{NES.name}")
        subrtn, subrtn_hparam = create_nes_subrtn(ex_dir, env_sim, nflow_policy)
    else:
        raise NotImplementedError("Select mode cem, reps, or nes via the command line argument -m")

    # Set the seed
    pyrado.set_seed(1001, verbose=True)

    # Set the hyper-parameters of StochSysIdViaNFlows
    num_eval_rollouts = 5
    algo_hparam = dict(
        metric=None,
        std_obs_filt=5,
        obs_dim_weight=[1, 1, 1, 1],  # bob setup
        # obs_dim_weight=[1, 1, 1, 1, 10, 10],   # qq-su setup
        num_rollouts_per_distr=len(dp_map) * 50,  # 3
        num_workers=subrtn_hparam["num_workers"],
    )

    # Save the environments and the hyper-parameters
    save_list_of_dicts_to_yaml(
        [
            dict(env=env_hparams),
            dict(subrtn=subrtn_hparam, subrtn_name=subrtn.name),
            dict(algo=algo_hparam, algo_name=StochSysIdViaNFlows.name, dp_map=dp_map),
        ],
        ex_dir,
    )

    algo = StochSysIdViaNFlows(subrtn, behavior_policy, **algo_hparam)

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
            f"{StochSysIdViaNFlows.name} finished training a {nflow_policy.name} "
            f"with {nflow_policy.num_param} parameters. {stopping_reason}",
            "g",
        )
    else:
        print_cbt(f"{subrtn.name} finished training. {stopping_reason}", "g")
