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
Simulate (with animation) a rollout in a live perturbed environment.
"""
import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import (
    DomainRandWrapperLive,
    DomainRandWrapper,
    DomainRandWrapperBuffer,
)
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Get the simulation environment
    env, policy, kwout = load_experiment(ex_dir, args)

    # Override the time step size if specified
    if args.dt is not None:
        env.dt = args.dt

    # Check which algorithm was used in the experiment
    algo = Algorithm.load_snapshot(load_dir=ex_dir, load_name="algo")

    if algo.name in ["npdr", "bayessim"]:
        # Sample domain parameters from the posterior, and
        domain_params, _ = NPDR.eval_posterior(
            kwout["posterior"],
            kwout["observations_real"],
            num_samples=args.num_samples,
            calculate_log_probs=False,
            normalize_posterior=False,
            subrtn_sbi_sampling_hparam=None,
        )
        env = DomainRandWrapperBuffer(env, randomizer=None, selection="random")
        NPDR.fill_domain_param_buffer(env, algo.dp_mapping, domain_params.squeeze(0))
        print_cbt("Using loaded randomizer obtained from posterior.", "c")

    elif not isinstance(env, DomainRandWrapper):
        # Add default domain randomization wrapper with action delay
        randomizer = create_default_randomizer(env)
        env = ActDelayWrapper(env)
        randomizer.add_domain_params(UniformDomainParam(name="act_delay", mean=5, halfspan=5, clip_lo=0, roundint=True))
        env = DomainRandWrapperLive(env, randomizer)
        print_cbt("Using default randomizer with additional action delay.", "c")

    else:
        print_cbt("Using loaded randomizer.", "c")

    # Simulate
    done, state, param = False, None, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=args.verbose, video=True),
            eval=True,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        print_domain_params(env.domain_param)
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
