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
Script to get the maximizer of a GP's posterior given saved data from a BayRn experiment
"""
import pyrado
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.episodic.nes import NES
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.step_based.ppo import PPO, PPO2
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the environment and the policy
    env_sim, policy, kwout = load_experiment(ex_dir, args)

    # Decide on which algorithm to use via the mode argument
    if args.mode == PPO.name:
        critic = GAE(kwout['value_fcn'], **kwout['hparams']['critic'])
        subroutine = PPO(ex_dir, env_sim, policy, critic, **kwout['hparams']['subrtn'])
    elif args.mode == PPO2.name:
        critic = GAE(kwout['value_fcn'], **kwout['hparams']['critic'])
        subroutine = PPO2(ex_dir, env_sim, policy, critic, **kwout['hparams']['subrtn'])
    elif args.mode == CEM.name:
        subroutine = CEM(ex_dir, env_sim, policy, **kwout['hparams']['subrtn'])
    elif args.mode == NES.name:
        subroutine = NES(ex_dir, env_sim, policy, **kwout['hparams']['subrtn'])
    elif args.mode == PoWER.name:
        subroutine = PoWER(ex_dir, env_sim, policy, **kwout['hparams']['subrtn'])
    else:
        raise NotImplementedError('Only PPO, PPO2, CEM, NES, and PoWER are implemented so far.')

    # Start from previous results policy if desired
    ppi = policy.param_values.data if args.warmstart is not None else None
    vpi = kwout['value_fcn'].param_values.data if args.warmstart is not None else None

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Train the policy on the most lucrative domain
    BayRn.train_argmax_policy(ex_dir, env_sim, subroutine, num_restarts=500, num_samples=1000,
                              policy_param_init=ppi, valuefcn_param_init=vpi)
