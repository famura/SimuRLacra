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
Script to run experiments on the Quanser Cart-Pole
"""
import joblib
import numpy as np
import os.path as osp

import pyrado
from pyrado.environments.quanser.base import QuanserReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleStabReal
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import ask_for_experiment, save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


def volt_disturbance_pos(t: float):
    return [6.0]


def volt_disturbance_neg(t: float):
    return [-6.0]


def experiment_wo_distruber(env_real: QuanserReal, env_sim: SimEnv):
    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt("Running the evaluation policy ...", "c", bright=True)
    return rollout(
        env_real,
        policy,
        eval=True,
        max_steps=args.max_steps,
        render_mode=RenderMode(text=True),
        no_reset=True,
        no_close=True,
    )


def experiment_w_distruber(env_real: QuanserReal, env_sim: SimEnv):
    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt("Running the evaluation policy ...", "c", bright=True)
    ro1 = rollout(
        env_real,
        policy,
        eval=True,
        max_steps=args.max_steps // 3,
        render_mode=RenderMode(),
        no_reset=True,
        no_close=True,
    )

    # Run disturber
    env_real = inner_env(env_real)  # since we are reusing it
    print_cbt("Running the 1st disturber ...", "c", bright=True)
    rollout(
        env_real,
        disturber_pos,
        eval=True,
        max_steps=steps_disturb,
        render_mode=RenderMode(),
        no_reset=True,
        no_close=True,
    )

    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt("Running the evaluation policy ...", "c", bright=True)
    ro2 = rollout(
        env_real,
        policy,
        eval=True,
        max_steps=args.max_steps // 3,
        render_mode=RenderMode(),
        no_reset=True,
        no_close=True,
    )

    # Run disturber
    env_real = inner_env(env_real)  # since we are reusing it
    print_cbt("Running the 2nd disturber ...", "c", bright=True)
    rollout(
        env_real,
        disturber_neg,
        eval=True,
        max_steps=steps_disturb,
        render_mode=RenderMode(),
        no_reset=True,
        no_close=True,
    )

    # Wrap the environment in the same as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run learned policy on the device
    print_cbt("Running the evaluation policy ...", "c", bright=True)
    ro3 = rollout(
        env_real,
        policy,
        eval=True,
        max_steps=args.max_steps // 3,
        render_mode=RenderMode(),
        no_reset=True,
        no_close=True,
    )

    return StepSequence.concat([ro1, ro2, ro3])


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(show_hyper_parameters=args.show_hparams) if args.dir is None else args.dir
    ex_tag = ex_dir.split("--", 1)[1]

    # Load the policy and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir)

    if args.verbose:
        print(f"Policy params:\n{policy.param_values.detach().cpu().numpy()}")

    # Create real-world counterpart (without domain randomization)
    env_real = QCartPoleStabReal(args.dt, args.max_steps)
    print_cbt("Set up the QCartPoleStabReal environment.", "c")

    # Set up the disturber
    disturber_pos = TimePolicy(env_real.spec, volt_disturbance_pos, env_real.dt)
    disturber_neg = TimePolicy(env_real.spec, volt_disturbance_neg, env_real.dt)
    steps_disturb = 10
    print_cbt(
        f"Set up the disturbers for the QCartPoleStabReal environment."
        f"\nVolt disturbance: {6} volts for {steps_disturb} steps",
        "c",
    )

    # Center cart and reset velocity filters and wait until the user or the conroller has put pole upright
    env_real.reset()
    print_cbt("Ready", "g")

    ros = []
    for r in range(args.num_runs):
        if args.mode == "wo":
            ro = experiment_wo_distruber(env_real, env_sim)
        elif args.mode == "w":
            ro = experiment_w_distruber(env_real, env_sim)
        else:
            raise pyrado.ValueErr(given=args.mode, eq_constraint="without (wo), or with (w) disturber")
        ros.append(ro)

    env_real.close()

    # Print and save results
    avg_return = np.mean([ro.undiscounted_return() for ro in ros])
    print_cbt(f"Average return: {avg_return}", "g", bright=True)
    save_dir = setup_experiment("evaluation", "qcp-st_experiment", ex_tag, base_dir=pyrado.TEMP_DIR)
    joblib.dump(ros, osp.join(save_dir, "experiment_rollouts.pkl"))
    save_dicts_to_yaml(
        dict(ex_dir=ex_dir, avg_return=avg_return, num_runs=len(ros), steps_disturb=steps_disturb),
        save_dir=save_dir,
        file_name="experiment_summary",
    )
