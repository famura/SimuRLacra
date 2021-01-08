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
Stabilize the ball at a random initial position and then sqitch to the learned policy to stabilize it in the center.
"""
import joblib
import numpy as np
import os.path as osp

import pyrado
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.policies.special.environment_specific import QBallBalancerPDCtrl
from pyrado.environment_wrappers.utils import inner_env
from pyrado.logger.experiment import ask_for_experiment, save_list_of_dicts_to_yaml, setup_experiment
from pyrado.sampling.rollout import rollout
from pyrado.spaces.polar import Polar2DPosSpace
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir
    ex_tag = ex_dir.split("--", 1)[1]

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir, args)

    if args.verbose:
        print(f"Policy params:\n{policy.param_values.detach().cpu().numpy()}")

    # Create real-world counterpart (without domain randomization)
    env_real = QBallBalancerReal(args.dt)
    print_cbt("Set up the QBallBalancerReal environment.", "c")

    # Set up PD-controller
    pdctrl = QBallBalancerPDCtrl(env_real.spec)
    if args.random_init_state:
        # Random initial state
        min_init_state = np.array([0.7 * 0.275 / 2, 0])
        max_init_state = np.array([0.8 * 0.275 / 2, 2 * np.pi])
        init_space = Polar2DPosSpace(min_init_state, max_init_state)
        init_state = init_space.sample_uniform()
    else:
        # Center of the plate is initial state
        # init_state = np.zeros(2)
        # init_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        # init_state = np.array([1., 0.])
        # init_state = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
        # init_state = np.array([0., -1.])
        # init_state = np.array([-1/np.sqrt(2), -1/np.sqrt(2)])
        # init_state = np.array([-1., 0.])
        # init_state = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])
        init_state = np.array([0.0, 1.0])
        init_state *= 0.103125  # distance scaling [m]
    pdctrl.reset(state_des=init_state)
    print_cbt(f"Set up the PD-controller for the QBallBalancerReal environment.\nDesired state: {init_state}", "c")

    ros = []
    for r in range(args.num_runs):
        # Run PD-controller on the device to get the ball into position
        env_real = inner_env(env_real)  # since we are reusing it
        print_cbt("Running the PD-controller ...", "c", bright=True)
        rollout(env_real, pdctrl, eval=True, max_steps=2000, render_mode=RenderMode())
        env_real.reset()

        # Wrap the real environment in the same way as done during training (do this after the PD controller finished)
        env_real = wrap_like_other_env(env_real, env_sim)

        # Run learned policy on the device
        print_cbt("Running the evaluation policy ...", "c", bright=True)
        ros.append(rollout(env_real, policy, eval=True, max_steps=args.max_steps, render_mode=RenderMode()))

    # Print and save results
    avg_return = np.mean([ro.undiscounted_return() for ro in ros])
    print_cbt(f"Average return: {avg_return}", "g", bright=True)
    save_dir = setup_experiment("evaluation", "qbb_experiment", ex_tag, base_dir=pyrado.TEMP_DIR)
    joblib.dump(ros, osp.join(save_dir, "experiment_rollouts.pkl"))
    save_list_of_dicts_to_yaml(
        [dict(ex_dir=ex_dir, init_state=init_state, avg_return=avg_return, num_runs=len(ros))],
        save_dir,
        file_name="experiment_summary",
    )

    # Stabilize at the end
    pdctrl.reset(state_des=np.zeros(2))
    rollout(env_real, pdctrl, eval=True, max_steps=1000, render_mode=RenderMode(text=True))
