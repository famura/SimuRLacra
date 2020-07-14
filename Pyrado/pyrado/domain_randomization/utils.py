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

import numpy as np
from copy import deepcopy
from typing import Sequence
from tabulate import tabulate

import pyrado


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


def print_domain_params(domain_params: [dict, Sequence[dict]]):
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
                print(tabulate([dp.values() for dp in domain_params],
                               headers=domain_params[0].keys(), tablefmt='simple'))
            else:
                raise pyrado.TypeErr(given=domain_params, expected_type=dict)

        elif isinstance(domain_params, dict):
            dp = deepcopy(domain_params)
            for k, v in dp.items():
                # Check if the values of the dirct are iterable
                if isinstance(v, (int, float, bool)):
                    dp[k] = [v]  # cast float to list of one element to make it iterable for tabulate
                if isinstance(v, np.ndarray) and v.size == 1:
                    dp[k] = [v.item()]  # cast scalar array to list of one element to make it iterable for tabulate
                elif isinstance(v, list):
                    pass
                else:
                    pyrado.TypeErr(given=v, expected_type=[int, float, bool, list])
            print(tabulate(dp, headers="keys", tablefmt='simple'))

        else:
            raise pyrado.TypeErr(given=domain_params, expected_type=[dict, list])
