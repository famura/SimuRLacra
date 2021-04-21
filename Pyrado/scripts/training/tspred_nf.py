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
Train a recurrent neural network to predict a time series of data
"""
import os.path as osp

import pandas as pd
import torch as to
import torch.nn as nn
import torch.optim as optim

import pyrado
from pyrado.algorithms.regression.timeseries_prediction import TSPred
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.recurrent.neural_fields import NFPolicy
from pyrado.spaces import BoxSpace
from pyrado.spaces.box import InfBoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_sets import TimeSeriesDataSet
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.functions import skyline


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Select the data set
    data_set_name = args.mode or "skyline"

    # Experiment
    ex_dir = setup_experiment(TSPred.name, NFPolicy.name)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Load the data
    if data_set_name == "skyline":
        dt = 0.01
        _, vals = skyline(
            dt=dt, t_end=20.0, t_intvl_space=BoxSpace(0.5, 3, shape=(1,)), val_space=BoxSpace(-2.0, 3.0, shape=(1,))
        )
        data = to.from_numpy(vals).view(-1, 1)
    else:
        data = pd.read_csv(osp.join(pyrado.PERMA_DIR, "time_series", f"{data_set_name}.csv"))
        if data_set_name == "daily_min_temperatures":
            data = to.tensor(data["Temp"].values, dtype=to.get_default_dtype()).view(-1, 1)
            dt = 1.0
        elif data_set_name == "monthly_sunspots":
            data = to.tensor(data["Sunspots"].values, dtype=to.get_default_dtype()).view(-1, 1)
            dt = 1.0
        elif "oscillation" in data_set_name:
            data = to.tensor(data["Positions"].values, dtype=to.get_default_dtype()).view(-1, 1)
            dt = 0.02
        else:
            raise pyrado.ValueErr(
                given=data_set_name,
                eq_constraint="'daily_min_temperatures', 'monthly_sunspots', "
                "'oscillation_50Hz_initpos-0.5', or 'oscillation_100Hz_initpos-0.4",
            )

    # Dataset
    data_set_hparam = dict(
        name=data_set_name, ratio_train=0.8, window_size=50, standardize_data=False, scale_min_max_data=True
    )
    dataset = TimeSeriesDataSet(data, **data_set_hparam)

    # Policy
    policy_hparam = dict(
        dt=dt,
        hidden_size=21,
        obs_layer=None,
        activation_nonlin=to.sigmoid,
        mirrored_conv_weights=True,
        conv_out_channels=1,
        conv_kernel_size=None,
        conv_padding_mode="circular",
        tau_init=10.0 if "oscillation" in data_set_name else 1.0,
        tau_learnable=True,
        kappa_init=1e-3,
        kappa_learnable=True,
        potential_init_learnable=True,
        # init_param_kwargs=dict(bell=True),
        use_cuda=False,
    )
    policy = NFPolicy(spec=EnvSpec(act_space=InfBoxSpace(shape=1), obs_space=InfBoxSpace(shape=1)), **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000,
        windowed=False,
        cascaded=True,
        optim_class=optim.Adam,
        optim_hparam=dict(lr=1e-2, eps=1e-8, weight_decay=1e-4),  # momentum=0.7
        loss_fcn=nn.MSELoss(),
    )
    algo = TSPred(ex_dir, dataset, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(data_set=data_set_hparam, data_set_name=data_set_name, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train()
