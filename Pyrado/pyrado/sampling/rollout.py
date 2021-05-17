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

import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch as to
import torch.nn as nn
from matplotlib import pyplot as plt
from tabulate import tabulate

import pyrado
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.utils import inner_env, typed_env
from pyrado.environments.base import Env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.curve import draw_dts
from pyrado.plotting.policy_parameters import draw_policy_params
from pyrado.plotting.rollout_based import (
    plot_actions,
    plot_features,
    plot_observations,
    plot_observations_actions_rewards,
    plot_potentials,
    plot_rewards,
    plot_states,
)
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import color_validity, print_cbt


def rollout(
    env: Env,
    policy: Union[nn.Module, Policy, Callable],
    eval: bool = False,
    max_steps: Optional[int] = None,
    reset_kwargs: Optional[dict] = None,
    render_mode: RenderMode = RenderMode(),
    render_step: int = 1,
    no_reset: bool = False,
    no_close: bool = False,
    record_dts: bool = False,
    stop_on_done: bool = True,
    seed: Optional[int] = None,
) -> StepSequence:
    """
    Perform a rollout (i.e. sample a trajectory) in the given environment using given policy.

    :param env: environment to use (`SimEnv` or `RealEnv`)
    :param policy: policy to determine the next action given the current observation.
                   This policy may be wrapped by an exploration strategy.
    :param eval: pass `False` if the rollout is executed during training, else `True`. Forwarded to PyTorch `Module`.
    :param max_steps: maximum number of time steps, if `None` the environment's property is used
    :param reset_kwargs: keyword arguments passed to environment's reset function
    :param render_mode: determines if the user sees an animation, console prints, or nothing
    :param render_step: rendering interval, renders every step if set to 1
    :param no_reset: do not reset the environment before running the rollout
    :param no_close: do not close (and disconnect) the environment after running the rollout
    :param record_dts: flag if the time intervals of different parts of one step should be recorded (for debugging)
    :param stop_on_done: set to false to ignore the environment's done flag (for debugging)
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return paths of the observations, actions, rewards, and information about the environment as well as the policy
    """
    # Check the input
    if not isinstance(env, Env):
        raise pyrado.TypeErr(given=env, expected_type=Env)
    # Don't restrain policy type, can be any callable
    if not isinstance(eval, bool):
        raise pyrado.TypeErr(given=eval, expected_type=bool)
    # The max_steps argument is checked by the environment's setter
    if not (isinstance(reset_kwargs, dict) or reset_kwargs is None):
        raise pyrado.TypeErr(given=reset_kwargs, expected_type=dict)
    if not isinstance(render_mode, RenderMode):
        raise pyrado.TypeErr(given=render_mode, expected_type=RenderMode)

    # Initialize the paths
    obs_hist = []
    act_hist = []
    act_app_hist = []
    rew_hist = []
    state_hist = []
    env_info_hist = []
    t_hist = []
    if isinstance(policy, Policy):
        if policy.is_recurrent:
            hidden_hist = []
        # If an ExplStrat is passed use the policy property, if a Policy is passed use it directly
        if isinstance(getattr(policy, "policy", policy), PotentialBasedPolicy):
            pot_hist = []
            stim_ext_hist = []
            stim_int_hist = []
        elif isinstance(getattr(policy, "policy", policy), TwoHeadedPolicy):
            head_2_hist = []
        if record_dts:
            dt_policy_hist = []
            dt_step_hist = []
            dt_remainder_hist = []

    # Override the number of steps to execute
    if max_steps is not None:
        env.max_steps = max_steps

    # Set all rngs' seeds (call before resetting)
    if seed is not None:
        pyrado.set_seed(seed)

    # Reset the environment and pass the kwargs
    if reset_kwargs is None:
        reset_kwargs = dict()
    obs = np.zeros(env.obs_space.shape) if no_reset else env.reset(**reset_kwargs)

    # Setup rollout information
    rollout_info = dict(env_name=env.name, env_spec=env.spec)
    if isinstance(inner_env(env), SimEnv):
        rollout_info["domain_param"] = env.domain_param

    if isinstance(policy, Policy):
        # Reset the policy, i.e. the exploration strategy in case of step-based exploration.
        # In case the environment is a simulation, the current domain parameters are passed to the policy. This allows
        # the policy policy to update it's internal model, e.g. for the energy-based swing-up controllers
        policy.reset(domain_param=rollout_info.get("domain_param", None))

        # Set dropout and batch normalization layers to the right mode
        if eval:
            policy.eval()
        else:
            policy.train()

        # Check for recurrent policy, which requires initializing the hidden state
        if policy.is_recurrent:
            hidden = policy.init_hidden()

    # Initialize animation
    env.render(render_mode, render_step=1)

    # Initialize the main loop variables
    done = False
    t = 0.0  # time starts at zero
    t_hist.append(t)
    if record_dts:
        t_post_step = time.time()  # first sample of remainder is useless

    # ----------
    # Begin loop
    # ----------

    # Terminate if the environment signals done, it also keeps track of the time
    while not (done and stop_on_done) and env.curr_step < env.max_steps:
        # Record step start time
        if record_dts or render_mode.video:
            t_start = time.time()  # dual purpose
        if record_dts:
            dt_remainder = t_start - t_post_step

        # Check observations
        if np.isnan(obs).any():
            env.render(render_mode, render_step=1)
            raise pyrado.ValueErr(
                msg=f"At least one observation value is NaN!"
                + tabulate(
                    [list(env.obs_space.labels), [*color_validity(obs, np.invert(np.isnan(obs)))]], headers="firstrow"
                )
            )

        # Get the agent's action
        obs_to = to.from_numpy(obs).type(to.get_default_dtype())  # policy operates on PyTorch tensors
        with to.no_grad():
            if isinstance(policy, Policy):
                if policy.is_recurrent:
                    if isinstance(getattr(policy, "policy", policy), TwoHeadedPolicy):
                        act_to, head_2_to, hidden_next = policy(obs_to, hidden)
                    else:
                        act_to, hidden_next = policy(obs_to, hidden)
                else:
                    if isinstance(getattr(policy, "policy", policy), TwoHeadedPolicy):
                        act_to, head_2_to = policy(obs_to)
                    else:
                        act_to = policy(obs_to)
            else:
                # If the policy ist not of type Policy, it should still operate on PyTorch tensors
                act_to = policy(obs_to)

        act = act_to.detach().cpu().numpy()  # environment operates on numpy arrays

        # Check actions
        if np.isnan(act).any():
            env.render(render_mode, render_step=1)
            raise pyrado.ValueErr(
                msg=f"At least one action value is NaN!"
                + tabulate(
                    [list(env.act_space.labels), [*color_validity(act, np.invert(np.isnan(act)))]], headers="firstrow"
                )
            )

        # Record time after the action was calculated
        if record_dts:
            t_post_policy = time.time()

        # Ask the environment to perform the simulation step
        state = env.state.copy()
        obs_next, rew, done, env_info = env.step(act)

        # Get the potentially clipped action, i.e. the one that was actually done in the environment
        act_app = env.limit_act(act)

        # Record time after the step i.e. the send and receive is completed
        if record_dts:
            t_post_step = time.time()
            dt_policy = t_post_policy - t_start
            dt_step = t_post_step - t_post_policy

        # Record data
        obs_hist.append(obs)
        act_hist.append(act)
        act_app_hist.append(act_app)
        rew_hist.append(rew)
        state_hist.append(state)
        env_info_hist.append(env_info)
        if record_dts:
            dt_policy_hist.append(dt_policy)
            dt_step_hist.append(dt_step)
            dt_remainder_hist.append(dt_remainder)
            t += dt_policy + dt_step + dt_remainder
        else:
            t += env.dt
        t_hist.append(t)
        if isinstance(policy, Policy):
            if policy.is_recurrent:
                hidden_hist.append(hidden)
                hidden = hidden_next
            # If an ExplStrat is passed use the policy property, if a Policy is passed use it directly
            if isinstance(getattr(policy, "policy", policy), PotentialBasedPolicy):
                pot_hist.append(hidden)
                stim_ext_hist.append(getattr(policy, "policy", policy).stimuli_external.detach().cpu().numpy())
                stim_int_hist.append(getattr(policy, "policy", policy).stimuli_internal.detach().cpu().numpy())
            elif isinstance(getattr(policy, "policy", policy), TwoHeadedPolicy):
                head_2_hist.append(head_2_to)

        # Store the observation for next step (if done, this is the final observation)
        obs = obs_next

        # Render if wanted (actually renders the next state)
        env.render(render_mode, render_step)
        if render_mode.video:
            do_sleep = True
            if pyrado.mujoco_loaded:
                from pyrado.environments.mujoco.base import MujocoSimEnv

                if isinstance(env, MujocoSimEnv):
                    # MuJoCo environments seem to crash on time.sleep()
                    do_sleep = False
            if do_sleep:
                # Measure time spent and sleep if needed
                t_end = time.time()
                t_sleep = env.dt + t_start - t_end
                if t_sleep > 0:
                    time.sleep(t_sleep)

    # --------
    # End loop
    # --------

    if not no_close:
        # Disconnect from EnvReal instance (does nothing for EnvSim instances)
        env.close()

    # Add final observation to observations list
    obs_hist.append(obs)
    state_hist.append(env.state.copy())

    # Return result object
    res = StepSequence(
        observations=obs_hist,
        actions=act_hist,
        actions_applied=act_app_hist,
        rewards=rew_hist,
        states=state_hist,
        time=t_hist,
        rollout_info=rollout_info,
        env_infos=env_info_hist,
        complete=True,  # the rollout function always returns complete paths
    )

    # Add special entries to the resulting rollout
    if isinstance(policy, Policy):
        if policy.is_recurrent:
            res.add_data("hidden_states", hidden_hist)
        if isinstance(getattr(policy, "policy", policy), PotentialBasedPolicy):
            res.add_data("potentials", pot_hist)
            res.add_data("stimuli_external", stim_ext_hist)
            res.add_data("stimuli_internal", stim_int_hist)
        elif isinstance(getattr(policy, "policy", policy), TwoHeadedPolicy):
            res.add_data("head_2", head_2_hist)
    if record_dts:
        res.add_data("dts_policy", dt_policy_hist)
        res.add_data("dts_step", dt_step_hist)
        res.add_data("dts_remainder", dt_remainder_hist)

    return res


def after_rollout_query(
    env: Env, policy: Policy, rollout: StepSequence
) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
    """
    Ask the user what to do after a rollout has been animated.

    :param env: environment used for the rollout
    :param policy: policy used for the rollout
    :param rollout: collected data from the rollout
    :return: done flag, initial state, and domain parameters
    """
    # Fist entry contains hotkey, second the info text
    options = [
        ["C", "continue simulation (with domain randomization)"],
        ["N", "set domain parameters to nominal values, and continue"],
        ["F", "fix the initial state"],
        ["I", "print information about environment (including randomizer), and policy"],
        ["S", "set a domain parameter explicitly"],
        ["P", "plot all observations, actions, and rewards"],
        ["PS [indices]", "plot all states, or selected ones by passing separated integers"],
        ["PO [indices]", "plot all observations, or selected ones by passing separated integers"],
        ["PA", "plot actions"],
        ["PR", "plot rewards"],
        ["PF", "plot features (for linear policy)"],
        ["PPOT", "plot potentials, stimuli, and actions (for potential-based policies)"],
        ["PDT", "plot time deltas (profiling of a real system)"],
        ["E", "exit"],
    ]

    # Ask for user input
    ans = input(tabulate(options, tablefmt="simple") + "\n").lower()

    if ans == "c" or ans == "":
        # We don't have to do anything here since the env will be reset at the beginning of the next rollout
        return False, None, None

    elif ans == "f":
        try:
            if isinstance(inner_env(env), RealEnv):
                raise pyrado.TypeErr(given=inner_env(env), expected_type=SimEnv)
            elif isinstance(inner_env(env), SimEnv):
                # Get the user input
                usr_inp = input(
                    f"Enter the {env.obs_space.flat_dim}-dim initial state "
                    f"(format: each dim separated by a whitespace):\n"
                )
                state = list(map(float, usr_inp.split()))
                if isinstance(state, list):
                    state = np.array(state)
                    if state.shape != env.obs_space.shape:
                        raise pyrado.ShapeErr(given=state, expected_match=env.obs_space)
                else:
                    raise pyrado.TypeErr(given=state, expected_type=list)
                return False, state, {}
        except (pyrado.TypeErr, pyrado.ShapeErr):
            return after_rollout_query(env, policy, rollout)

    elif ans == "n":
        # Get nominal domain parameters
        if isinstance(inner_env(env), SimEnv):
            dp_nom = inner_env(env).get_nominal_domain_param()
            if typed_env(env, ActDelayWrapper) is not None:
                # There is an ActDelayWrapper in the env chain
                dp_nom["act_delay"] = 0
        else:
            dp_nom = None
        return False, None, dp_nom

    elif ans == "i":
        # Print the information and return to the query
        print(env)
        if hasattr(env, "randomizer"):
            print(env.randomizer)
        print(policy)
        return after_rollout_query(env, policy, rollout)

    elif ans == "p":
        plot_observations_actions_rewards(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pa":
        plot_actions(rollout, env)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "po":
        plot_observations(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif "po" in ans and any(char.isdigit() for char in ans):
        idcs = [int(s) for s in ans.split() if s.isdigit()]
        plot_observations(rollout, idcs_sel=idcs)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "ps":
        plot_states(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif "ps" in ans and any(char.isdigit() for char in ans):
        idcs = [int(s) for s in ans.split() if s.isdigit()]
        plot_states(rollout, idcs_sel=idcs)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pf":
        plot_features(rollout, policy)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pp":
        draw_policy_params(policy, env.spec, annotate=False)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pr":
        plot_rewards(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "pdt":
        draw_dts(rollout.dts_policy, rollout.dts_step, rollout.dts_remainder)
        plt.show()
        return (after_rollout_query(env, policy, rollout),)

    elif ans == "ppot":
        plot_potentials(rollout)
        plt.show()
        return after_rollout_query(env, policy, rollout)

    elif ans == "s":
        if isinstance(env, SimEnv):
            dp = env.get_nominal_domain_param()
            for k, v in dp.items():
                dp[k] = [v]  # cast float to list of one element to make it iterable for tabulate
            print("These are the nominal domain parameters:")
            print(tabulate(dp, headers="keys", tablefmt="simple"))

        # Get the user input
        strs = input("Enter one new domain parameter\n(format: key whitespace value):\n")
        try:
            param = dict(str.split() for str in strs.splitlines())
            # Cast the values of the param dict from str to float
            for k, v in param.items():
                param[k] = float(v)
            return False, None, param
        except (ValueError, KeyError):
            print_cbt(f"Could not parse {strs} into a dict.", "r")
            after_rollout_query(env, policy, rollout)

    elif ans == "e":
        env.close()
        return True, None, {}  # breaks the outer while loop

    else:
        return after_rollout_query(env, policy, rollout)  # recursion
