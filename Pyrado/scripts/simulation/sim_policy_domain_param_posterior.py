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
Simulate (with animation) a rollout with domain parameters drawn from a posterior distribution obtained from running
Neural Posterior Domain Randomization or BayesSim.
"""
import operator

import prettyprinter
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.set_defaults(animation=True)  # different default value for this script
    args = parser.parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the experiment
    env, policy, kwout = load_experiment(ex_dir, args)
    env_real = pyrado.load("env_real.pkl", ex_dir)
    data_real = kwout["data_real"]
    if args.iter == -1:
        # This script is not made to evaluate multiple iterations at once, thus we always select the data one iteration
        data_real = to.atleast_2d(data_real[args.iter])

    # Override the time step size if specified
    if args.dt is not None:
        env.dt = args.dt

    # Use the environments number of steps in case of the default argument (inf)
    max_steps = env.max_steps if args.max_steps == pyrado.inf else args.max_steps

    # Check which algorithm was used in the experiment
    algo = Algorithm.load_snapshot(load_dir=ex_dir, load_name="algo")
    if not isinstance(algo, (NPDR, BayesSim)):
        raise pyrado.TypeErr(given=algo, expected_type=(NPDR, BayesSim))

    # Sample domain parameters from the posterior. Use all samples, by hijacking the get_ml_posterior_samples to obtain
    # them sorted.
    domain_params, log_probs = SBIBase.get_ml_posterior_samples(
        dp_mapping=algo.dp_mapping,
        posterior=kwout["posterior"],
        data_real=data_real,
        num_eval_samples=args.num_samples,
        num_ml_samples=args.num_samples,
        calculate_log_probs=True,
        normalize_posterior=args.normalize,
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
        return_as_tensor=False,
    )
    assert len(domain_params) == 1  # the list has as many elements as evaluated iterations
    domain_params = domain_params[0]

    if args.normalize:
        # If the posterior is normalized, we do not rescale the probabilities since they already sum to 1
        probs = to.exp(log_probs)
    else:
        # If the posterior is not normalized, we rescale the probabilities to make them interpretable
        probs = to.exp(log_probs - log_probs.max())  # scale the probabilities to [0, 1]
    probs = probs.T

    # Select edge cases
    idcs_sel = (0, 1, 2, -3, -2, -1)
    assert len(idcs_sel) <= args.num_samples
    print_cbt(f"Selected the indices {idcs_sel}", "c", bright=True)
    domain_params = operator.itemgetter(*idcs_sel)(domain_params)
    probs = operator.itemgetter(*idcs_sel)(probs)

    if isinstance(env_real, SimEnv):
        # Replay the ground truth environment
        done = False
        while not done:
            ro = rollout(env_real, policy, render_mode=RenderMode(video=args.animation, render=args.render), eval=True)
            print_cbt(f"Return: {ro.undiscounted_return()} in the ground truth environment", "g", bright=True)
            done = input("Repeat rollout? [y / any other] ").lower() != "y"

    # Simulate
    normalized_str = "(normalized)" if args.normalize else "(rescaled)"
    for domain_param, prob in zip(domain_params, probs):
        done = False
        while not done:
            # Get one fixed initial state to make them comparable
            init_state = env.init_space.sample_uniform()

            ro = rollout(
                env,
                policy,
                render_mode=RenderMode(video=args.animation, render=args.render),
                eval=True,
                reset_kwargs=dict(domain_param=domain_param, init_state=init_state),
            )
            print_cbt(
                f"Return: {ro.undiscounted_return()} with domain parameters sampled with "
                f"{normalized_str} probability {prob.numpy()}",
                "g",
                bright=True,
            )
            prettyprinter.pprint(domain_param)

            done = input("Repeat rollout? [y / any other] ").lower() != "y"
