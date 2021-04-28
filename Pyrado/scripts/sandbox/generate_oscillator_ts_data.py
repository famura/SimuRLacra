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
Script to generate time series data sets.
"""
import argparse
import functools
import os.path as osp
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

import pyrado
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.policies.feed_forward.time import TimePolicy
from pyrado.sampling.rollout import rollout


def _dirac_impulse(t, env_spec, amp) -> Callable:
    return amp * np.ones(env_spec.act_space.shape) if t == 0 else np.zeros(env_spec.act_space.shape)


def generate_oscillation_data(dt, t_end, excitation):
    """
    Use OMOEnv to generate a 1-dim damped oscillation signal.

    :param dt: time step size [s]
    :param t_end: Time duration [s]
    :param excitation: type of excitation, either (initial) 'position' or 'force' (function of time)
    :return: 1-dim oscillation trajectory
    """
    env = OneMassOscillatorSim(dt, np.ceil(t_end / dt))
    env.domain_param = dict(m=1.0, k=10.0, d=2.0)
    if excitation == "force":
        policy = TimePolicy(env.spec, functools.partial(_dirac_impulse, env_spec=env.spec, amp=0.5), dt)
        reset_kwargs = dict(init_state=np.array([0, 0]))
    elif excitation == "position":
        policy = IdlePolicy(env.spec)
        reset_kwargs = dict(init_state=np.array([0.5, 0]))
    else:
        raise pyrado.ValueErr(given=excitation, eq_constraint="'force' or 'position'")

    # Generate the data
    ro = rollout(env, policy, reset_kwargs=reset_kwargs, record_dts=False)
    return ro.observations[:, 0]


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "excitation", type=str, nargs="?", default="position", help="Type of excitation ('position' or 'force')"
    )
    parser.add_argument("dt", type=float, nargs="?", default="0.01", help="Time step size [s]")
    parser.add_argument("t_end", type=float, nargs="?", default="5.", help="Time duration [s]")
    parser.add_argument("std", type=float, nargs="?", default="0.02", help="Standard deviation of the noise")
    args = parser.parse_args()

    # Generate ground truth data
    data_gt = generate_oscillation_data(0.01, 5, args.excitation)

    # Add noise
    noise = np.random.randn(*data_gt.shape) * args.std
    data_n = data_gt + noise

    # Plot the data
    plt.plot(data_n, label="signal")
    plt.plot(data_gt, lw=3, label="ground truth")
    plt.legend()
    plt.show()

    # Save the data
    np.save(osp.join(pyrado.PERMA_DIR, "time_series", "omo_traj_gt.npy"), data_gt)
    np.save(osp.join(pyrado.PERMA_DIR, "time_series", "omo_traj_n.npy"), data_n)
