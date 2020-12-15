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
import torch.nn as nn

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy
from pyrado.policies.features import FeatureStack
from pyrado.policies.initialization import init_param


class LinearPolicy(Policy):
    """
    A linear policy defined by the inner product of nonlinear features of the observations with the policy parameters
    """

    name: str = "lin"

    def __init__(self, spec: EnvSpec, feats: FeatureStack, init_param_kwargs: dict = None, use_cuda: bool = False):
        """
        Constructor

        :param spec: specification of environment
        :param feats: list of feature functions
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        """
        if not isinstance(feats, FeatureStack):
            raise pyrado.TypeErr(given=feats, expected_type=FeatureStack)

        # Call Policy's constructor
        super().__init__(spec, use_cuda)

        self._feats = feats
        self.num_active_feat = feats.get_num_feat(spec.obs_space.flat_dim)
        self.net = nn.Linear(self.num_active_feat, spec.act_space.flat_dim, bias=False)

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    @property
    def features(self) -> FeatureStack:
        """ Get the (nonlinear) feature transformations. """
        return self._feats

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize the linear layer using default initialization
            init_param(self.net, **kwargs)
        else:
            self.param_values = init_values  # ignore the IntelliJ warning

    def eval_feats(self, obs: to.Tensor) -> to.Tensor:
        """
        Evaluate the features for the given observations.

        :param obs: observation from the environment
        :return feats_val: the features' values
        """
        return self._feats(obs)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Evaluate the features at the given observation or use given feature values

        :param obs: observations from the environment
        :return: actions
        """
        obs = obs.to(self.device)
        batched = obs.ndimension() == 2  # number of dim is 1 if unbatched, dim > 2 is cought by features
        feats_val = self.eval_feats(obs)

        # Inner product between policy parameters and the value of the features
        act = self.net(feats_val)

        # Return the flattened tensor if not run in a batch mode to be compatible with the action spaces
        return act.flatten() if not batched else act
