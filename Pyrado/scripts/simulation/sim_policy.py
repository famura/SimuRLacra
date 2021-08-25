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
Simulate (with animation) a rollout in an environment.
"""
from prettyprinter import pprint

import pyrado
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import (
    DomainRandWrapper,
    DomainRandWrapperLive,
    remove_all_dr_wrappers,
)
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.set_defaults(animation=True)  # different default value for this script
    args = parser.parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the environment and the policy
    env, policy, kwout = load_experiment(ex_dir, args)

    if args.verbose:
        print("Hyper-parameters of the experiment")
        pprint(kwout.get("hparams", "No hyper-parameters found!"))

    # Override the time step size if specified
    if args.dt is not None:
        env.dt = args.dt

    # Use the environments number of steps in case of the default argument (inf)
    max_steps = env.max_steps if args.max_steps == pyrado.inf else args.max_steps

    # Domain randomization
    if args.no_dr:
        env = remove_all_dr_wrappers(env, verbose=True)
        print_cbt("Simulating without domain randomization.", "c")
    else:
        if not isinstance(env, DomainRandWrapper):
            if input("No randomizer found. Shall one be added? [y / any other] ").lower() == "y":
                # Add default domain randomization wrapper with action delay
                randomizer = create_default_randomizer(env)
                env = ActDelayWrapper(env)
                randomizer.add_domain_params(
                    UniformDomainParam(name="act_delay", mean=5, halfspan=5, clip_lo=0, roundint=True)
                )
                env = DomainRandWrapperLive(env, randomizer)
                print_cbt("Using default randomizer with additional action delay.", "c")
            else:
                print_cbt("Simulating without domain randomization.", "c")
        else:
            print_cbt("Using loaded randomizer.", "c")

    # Simulate
    done, state, param = False, None, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=args.verbose, video=args.animation, render=args.render),
            eval=True,
            max_steps=max_steps,
            stop_on_done=not args.relentless,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        print_domain_params(env.domain_param)
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
