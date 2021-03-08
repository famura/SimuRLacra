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
Load and run a policy on the associated real-world Quanser environment
"""
import pyrado
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environment_wrappers.utils import inner_env
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(show_hyper_parameters=args.show_hparams) if args.dir is None else args.dir

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir, args)

    # Detect the correct real-world counterpart and create it
    if isinstance(inner_env(env_sim), QBallBalancerSim):
        env_real = QBallBalancerReal(dt=args.dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QCartPoleSim):
        env_real = QCartPoleReal(dt=args.dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QQubeSim):
        env_real = QQubeReal(dt=args.dt, max_steps=args.max_steps)
    else:
        raise pyrado.TypeErr(given=env_sim, expected_type=[QBallBalancerSim, QCartPoleSim, QQubeSim])

    # Wrap the real environment in the same way as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done = False
    print_cbt("Running loaded policy ...", "c", bright=True)
    while not done:
        ro = rollout(
            env_real, policy, eval=True, record_dts=True, render_mode=RenderMode(text=False, video=args.animation)
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, _, _ = after_rollout_query(env_real, policy, ro)
