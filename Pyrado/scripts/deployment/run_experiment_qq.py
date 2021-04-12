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
Script to run experiments on the Quanser Qube
"""
import os
import os.path as osp
import torch as to
from datetime import datetime

import pyrado
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.experiments import load_experiment
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir, args)
    if "argmax" in args.policy_name:
        policy = to.load(osp.join(ex_dir, "policy_argmax.pt"))
        print_cbt(f"Loaded {osp.join(ex_dir, 'policy_argmax.pt')}", "g", bright=True)

    # Create real-world counterpart
    max_steps = args.max_steps if args.max_steps < pyrado.inf else env_sim.max_steps
    dt = args.dt if args.dt is not None else env_sim.dt
    env_real = QQubeSwingUpReal(dt, max_steps)
    print_cbt(f"Set up the QQubeSwingUpReal environment with dt={env_real.dt} max_steps={env_real.max_steps}.", "c")

    # Finally wrap the env in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    ex_ts = datetime.now().strftime(pyrado.timestamp_format)
    save_dir = osp.join(ex_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    num_rollouts_per_config = args.num_rollouts_per_config if args.num_rollouts_per_config is not None else 5
    est_ret = BayRn.eval_policy(
        save_dir, env_real, policy, mc_estimator=True, prefix=ex_ts, num_rollouts=num_rollouts_per_config
    )
    print_cbt(f"Estimated return: {est_ret.item()}", "g")
