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
Run separate evaluation for comparing against BayRn
"""
import os
import os.path as osp
from datetime import datetime

from pyrado.algorithms.bayrn import BayRn
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.logger.experiment import ask_for_experiment, timestamp_format
from pyrado.utils.experiments import wrap_like_other_env, load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir)

    # Create real-world counterpart (without domain randomization)
    env_real = QQubeReal(env_sim.dt, env_sim.max_steps)
    print_cbt(f'Set up the QQubeReal environment with dt={env_real.dt} max_steps={env_real.max_steps}.', 'c')
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run the policy on the real system
    ex_ts = datetime.now().strftime(timestamp_format)
    save_dir = osp.join(ex_dir, 'evaluation')
    os.makedirs(save_dir, exist_ok=True)
    est_ret = BayRn.eval_policy(save_dir, env_real, policy, montecarlo_estimator=True, prefix=ex_ts, num_rollouts=5)

    print_cbt(f'Estimated return: {est_ret.item()}', 'g')
