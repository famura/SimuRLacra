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
Train an agent to solve the discrete Ball-on-Beam environment using Deep Q-Leaning.

.. note::
    The hyper-parameters are not tuned at all!
"""
import torch as to

import pyrado
from pyrado.algorithms.step_based.dql import DQL
from pyrado.environments.pysim.ball_on_beam import BallOnBeamDiscSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNN, DiscreteActQValPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment
    ex_dir = setup_experiment(BallOnBeamDiscSim.name, f"{DQL.name}_{DiscreteActQValPolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=500)
    env = BallOnBeamDiscSim(**env_hparams)

    # Policy
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    net = FNN(
        input_size=DiscreteActQValPolicy.get_qfcn_input_size(env.spec),
        output_size=DiscreteActQValPolicy.get_qfcn_output_size(),
        **policy_hparam,
    )
    policy = DiscreteActQValPolicy(spec=env.spec, net=net)

    # Algorithm
    algo_hparam = dict(
        max_iter=5000,
        memory_size=10 * env.max_steps,
        eps_init=0.1286,
        eps_schedule_gamma=0.9955,
        gamma=0.998,
        target_update_intvl=5,
        num_updates_per_step=20,
        max_grad_norm=0.5,
        min_steps=10,
        batch_size=256,
        num_workers=4,
        lr=7e-4,
    )
    algo = DQL(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="best", seed=args.seed)
