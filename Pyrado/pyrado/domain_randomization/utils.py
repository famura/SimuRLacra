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

from copy import deepcopy
from typing import Sequence, Union

import numpy as np
from tabulate import tabulate

import pyrado
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper, ObsRunningNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.utils.input_output import print_cbt


def param_grid(param_values: dict) -> list:
    """
    Create a parameter set for every possible combination of parameters.

    :param param_values: dict from parameter names to values for these parameters
    :return: list of parameter sets
    """
    # Create a meshgrid of the param values
    mg = np.meshgrid(*param_values.values())
    if not isinstance(mg, (list, np.ndarray)):
        raise pyrado.TypeErr(given=mg, expected_type=[list, np.ndarray])

    # Flatten the grid arrays so they can be iterated
    mg_flat = (arr.flatten() for arr in mg)

    # Convert the meshgrid arrays to a parameter set list
    return [dict(zip(param_values.keys(), pvals)) for pvals in zip(*mg_flat)]


def print_domain_params(domain_params: Union[dict, Sequence[dict]]):
    """
    Print a list of (domain parameter) dicts / a dict (of domain parameters) prettily.

    :param domain_params: list of dicts or a single dict containing the a list of domain parameters
    """
    if domain_params:
        # Do nothing if domain_param list/dict is empty

        if isinstance(domain_params, list):
            # Check the first element
            if isinstance(domain_params[0], dict):
                # Assuming all dicts have identical keys
                print(
                    tabulate([dp.values() for dp in domain_params], headers=domain_params[0].keys(), tablefmt="simple")
                )
            else:
                raise pyrado.TypeErr(given=domain_params, expected_type=dict)

        elif isinstance(domain_params, dict):
            dp = deepcopy(domain_params)
            for k, v in dp.items():
                if isinstance(v, list):
                    dp[k] = [float(i) for i in v]
                else:
                    try:
                        dp[k] = [float(v)]
                    except (ValueError, TypeError):
                        # noinspection PyBroadException
                        try:
                            dp[k] = v.tolist()  # numpy arrays and PyTorch tensors have a tolist() method
                        except Exception as ex:  # pylint: disable=broad-except
                            raise pyrado.TypeErr(
                                msg="The domain param entries need to either be a float, a numpy array or a"
                                "PyTorch tensor, such that they can be converted to a list!"
                            ) from ex
            # Taubulate is iterating through the lists in the dp dict
            print(tabulate(dp, headers="keys", tablefmt="simple"))

        else:
            raise pyrado.TypeErr(given=domain_params, expected_type=[dict, list])


def wrap_like_other_env(
    env_targ: Union[SimEnv, RealEnv], env_src: [SimEnv, EnvWrapper], use_downsampling: bool = False
) -> Union[SimEnv, RealEnv]:
    """
    Wrap a given real environment like it's simulated counterpart (except the domain randomization of course).

    :param env_targ: target environment e.g. environment representing the physical device
    :param env_src: source environment e.g. simulation environment used for training
    :param use_downsampling: apply a wrapper that downsamples the actions if the sampling frequencies don't match
    :return: target environment
    """
    if use_downsampling and env_src.dt > env_targ.dt:
        if typed_env(env_targ, DownsamplingWrapper) is None:
            ds_factor = int(env_src.dt / env_targ.dt)
            env_targ = DownsamplingWrapper(env_targ, ds_factor)
            print_cbt(f"Wrapped the target environment with a DownsamplingWrapper of factor {ds_factor}.", "y")
        else:
            print_cbt("The target environment was already wrapped with a DownsamplingWrapper.", "y")

    if typed_env(env_src, ActNormWrapper) is not None:
        if typed_env(env_targ, ActNormWrapper) is None:
            env_targ = ActNormWrapper(env_targ)
            print_cbt("Wrapped the target environment with an ActNormWrapper.", "y")
        else:
            print_cbt("The target environment was already wrapped with an ActNormWrapper.", "y")

    if typed_env(env_src, ObsNormWrapper) is not None:
        if typed_env(env_targ, ObsNormWrapper) is None:
            env_targ = ObsNormWrapper(env_targ)
            print_cbt("Wrapped the target environment with an ObsNormWrapper.", "y")
        else:
            print_cbt("The target environment was already wrapped with an ObsNormWrapper.", "y")

    if typed_env(env_src, ObsRunningNormWrapper) is not None:
        if typed_env(env_targ, ObsRunningNormWrapper) is None:
            env_targ = ObsRunningNormWrapper(env_targ)
            print_cbt("Wrapped the target environment with an ObsRunningNormWrapper.", "y")
        else:
            print_cbt("The target environment was already wrapped with an ObsRunningNormWrapper.", "y")

    if typed_env(env_src, ObsPartialWrapper) is not None:
        if typed_env(env_targ, ObsPartialWrapper) is None:
            env_targ = ObsPartialWrapper(
                env_targ, mask=typed_env(env_src, ObsPartialWrapper).keep_mask, keep_selected=True
            )
            print_cbt("Wrapped the target environment with an ObsPartialWrapper.", "y")
        else:
            print_cbt("The target environment was already wrapped with an ObsPartialWrapper.", "y")

    return env_targ
