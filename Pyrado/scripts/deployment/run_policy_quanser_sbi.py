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
Load and run a policy learned with NPDR or BayesSim on the associated real-world Quanser environment
"""
import pyrado
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSim
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleReal
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--init",
        action="store_true",
        default=False,
        help="set flag to load the initial (default) policy of the algorithm",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        default=False,
        help="use new sampled domain parameter for each rollout (only for 'prior' and 'posterior')",
    )
    parser.add_argument(
        "--src_domain_param",
        type=str,
        default=None,
        help="set the source of the policy's domain parameter ('ml', 'nominal', 'posterior', 'prior')",
    )
    args = parser.parse_args()

    # Check arguments
    src_domain_param_args = ["ml", "nominal", "posterior", "prior", None]
    if args.src_domain_param not in src_domain_param_args:
        raise pyrado.ValueErr(given_name="src_domain_param", eq_constraint=src_domain_param_args)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    if args.iter != -1:
        args.policy_name = f"iter_{args.iter}_policy"
    if args.init:
        args.policy_name = "init_policy"
    env_sim, policy, extra = load_experiment(ex_dir, args)

    # Create the domain parameter mapping
    dp_mapping = dict()
    if extra is not None:
        dp_counter = 0
        for key in sorted(extra["hparams"]["dp_mapping"].keys()):
            dp = extra["hparams"]["dp_mapping"][key]
            if dp in extra["hparams"]["dp_selection"]:
                dp_mapping[dp_counter] = dp
                dp_counter += 1

    pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=policy)

    # Reset the policy's domain parameter if desired
    prior, posterior = None, None
    if args.src_domain_param == "ml":
        ml_domain_param = pyrado.load("ml_domain_param.pkl", ex_dir, prefix=f"iter_{args.iter}")
        policy.reset(**dict(domain_param=ml_domain_param))
    elif args.src_domain_param == "posterior":
        prefix_str = "" if args.iter == -1 and args.round == -1 else f"iter_{args.iter}_round_{args.round}"
        posterior = pyrado.load("posterior.pt", ex_dir, prefix=prefix_str)
    elif args.src_domain_param == "prior":
        prior = pyrado.load("prior.pt", ex_dir)
    elif args.src_domain_param == "nominal":
        policy.reset(**dict(domain_param=env_sim.get_nominal_domain_param()))

    # Detect the correct real-world counterpart and create it
    if isinstance(inner_env(env_sim), QBallBalancerSim):
        env_real = QBallBalancerReal(dt=args.dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QCartPoleSim):
        env_real = QCartPoleReal(dt=args.dt, max_steps=args.max_steps)
    elif isinstance(inner_env(env_sim), QQubeSim):
        env_real = QQubeSwingUpReal(dt=args.dt, max_steps=args.max_steps)
    else:
        raise pyrado.TypeErr(given=env_sim, expected_type=[QBallBalancerSim, QCartPoleSim, QQubeSim])

    # Wrap the real environment in the same way as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done, first_round = False, True
    print_cbt("Running loaded policy ...", "c", bright=True)
    while not done:
        # sample new domain parameter
        if (args.resample or first_round) and args.src_domain_param in ["posterior", "prior"]:
            samples = None
            if args.src_domain_param == "prior":
                samples = prior.sample((1,)).flatten()
            elif args.src_domain_param == "posterior":
                samples = posterior.sample((1,), sample_with_mcmc=args.use_mcmc).flatten()
            dp = dict(zip([dp_mapping[k] for k in sorted(dp_mapping.keys())], samples.tolist()))
            print("Domain parameter:", *[f"\n\t{k}:\t{v}" for k, v in dp.items()])
            policy.reset(**dict(domain_param=dp))
            first_round = False

        ro = rollout(env_real, policy, eval=True, record_dts=True)
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, _, _ = after_rollout_query(env_real, policy, ro)
