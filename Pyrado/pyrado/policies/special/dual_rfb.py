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

import torch as to
from torch import nn as nn

import pyrado
from pyrado.policies.features import RBFFeat, FeatureStack
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.utils.data_types import EnvSpec


class DualRBFLinearPolicy(LinearPolicy):
    """
    A linear policy with RBF features which are also used to get the derivative of the features. The use-case in mind
    is a simple policy which generates the joint position and joint velocity commands for the internal PD-controller
    of a robot (e.g. Barrett WAM). By re-using the RBF, we reduce the number of parameters, while we can at the same
    time get the velocity information from the features, i.e. the derivative of the normalized Gaussians.
    """

    name: str = "dualrbf"

    def __init__(
        self, spec: EnvSpec, rbf_hparam: dict, dim_mask: int = 2, init_param_kwargs: dict = None, use_cuda: bool = False
    ):
        """
        Constructor

        :param spec: specification of environment
        :param rbf_hparam: hyper-parameters for the RBF-features, see `RBFFeat`
        :param dim_mask: number of RBF features to mask out at the beginning and the end of every dimension,
                         pass 1 to remove the first and the last features for the policy, pass 0 to use all
                         RBF features. Masking out RBFs makes sense if you want to obtain a smooth starting behavior.
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not spec.act_space.flat_dim % 2 == 0:
            raise pyrado.ShapeErr(
                msg="DualRBFLinearPolicy only works with an even number of actions, since we are using the time "
                "derivative of the features to create the second half of the outputs. This is done to use "
                "forward() in order to obtain the joint position and the joint velocities. Check the action space "
                "of the environment if the second halt of the actions space are velocities!"
            )
        if not (0 <= dim_mask <= rbf_hparam["num_feat_per_dim"] // 2):
            raise pyrado.ValueErr(
                given=dim_mask, ge_constraint="0", le_constraint=f"{rbf_hparam['num_feat_per_dim']//2}"
            )

        # Construct the RBF features
        self._feats = RBFFeat(**rbf_hparam)

        # Call LinearPolicy's constructor (custom parts will be overridden later)
        super().__init__(spec, FeatureStack([self._feats]), init_param_kwargs, use_cuda)

        # Override custom parts
        self._feats = RBFFeat(**rbf_hparam)
        self.dim_mask = dim_mask
        if self.dim_mask > 0:
            self.num_active_feat = self._feats.num_feat - 2 * self.dim_mask * spec.obs_space.flat_dim
        else:
            self.num_active_feat = self._feats.num_feat
        self.net = nn.Linear(self.num_active_feat, self.env_spec.act_space.flat_dim // 2, bias=False)

        # Create mask to deactivate first and last feature of every input dimension
        self.feats_mask = to.ones(self._feats.centers.shape, dtype=to.bool)
        self.feats_mask[: self.dim_mask, :] = False
        self.feats_mask[-self.dim_mask :, :] = False
        self.feats_mask = self.feats_mask.t().reshape(-1)  # reshape the same way as in RBFFeat

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Evaluate the features at the given observation or use given feature values

        :param obs: observations from the environment
        :return: actions
        """
        obs = obs.to(self.device)
        batched = obs.ndimension() == 2  # number of dim is 1 if unbatched, dim > 2 is caught by features
        feats_val = self._feats(obs)
        feats_dot = self._feats.derivative(obs)

        if self.dim_mask > 0:
            # Mask out first and last feature of every input dimension
            feats_val = feats_val[:, self.feats_mask]
            feats_dot = feats_dot[:, self.feats_mask]

        # Inner product between policy parameters and the value of the features
        act_pos = self.net(feats_val)
        act_vel = self.net(feats_dot)
        act = to.cat([act_pos, act_vel], dim=1)

        # Return the flattened tensor if not run in a batch mode to be compatible with the action spaces
        return act.flatten() if not batched else act
