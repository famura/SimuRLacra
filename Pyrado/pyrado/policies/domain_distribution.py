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
import torch as to
from math import sqrt
from typing import Dict, Tuple
from torch import nn as nn

import pyrado
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.policies.base import Policy
from pyrado.spaces import BoxSpace
from pyrado.spaces.empty import EmptySpace
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import clamp


class DomainDistrParamPolicy(Policy):
    """ A proxy to the Policy class in order to use the policy's parameters as domain distribution parameters """

    name: str = 'ddp'
    min_dp_var: float = 1e-6  # hard coded lower bound on all domain parameters variances

    def __init__(self,
                 mapping: Dict[int, Tuple[str, str]],
                 trafo_mask: list,
                 prior: DomainRandomizer = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param mapping: mapping from index of the numpy array (coming from the algorithm) to domain parameter name
                        (e.g. mass, length) and the domain distribution parameter (e.g. mean, std)
        :param trafo_mask: every domain distribution parameter that is set to `True` in this mask will be exponentially
                           transformed. This is useful to avoid setting negative variance.
        :param prior: prior believe about the distribution parameters in from of a `DomainRandomizer`
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(mapping, dict):
            raise pyrado.TypeErr(given=mapping, expected_type=dict)
        if not len(trafo_mask) == len(mapping):
            raise pyrado.ShapeErr(given=trafo_mask, expected_match=mapping)

        self.mapping = mapping
        self.mask = to.tensor(trafo_mask, dtype=to.bool)

        # Construct a valid space for the policy parameters aka domain distribution parameters
        bound_lo = -pyrado.inf*np.ones(len(mapping))
        bound_up = +pyrado.inf*np.ones(len(mapping))
        lables = len(mapping)*['None']
        # for idx, ddp in self.mapping.items():
        #     lables[idx] = f'{ddp[0]}_{ddp[1]}'
        #     if ddp[1] == 'std':  # 2nd output is the name of the domain distribution param
        #         bound_lo[idx] = sqrt(DomainDistrParamPolicy.min_dp_var)
        #     elif ddp[1] == 'halfspan':  # 2nd output is the name of the domain distribution param
        #         bound_lo[idx] = sqrt(12*DomainDistrParamPolicy.min_dp_var)/2  # var(U) = (2*halfspan)^2/12

        # Define the parameter space by using the Policy.env_spec.act_space
        param_spec = EnvSpec(obs_space=EmptySpace(), act_space=BoxSpace(bound_lo, bound_up, labels=lables))

        # Call Policy's constructor
        super().__init__(param_spec, use_cuda)

        self.params = nn.Parameter(to.Tensor(param_spec.act_space.flat_dim), requires_grad=True)
        self.prior = prior
        self.init_param(prior=prior)

    def masked_exp_transform(self, params: to.Tensor) -> to.Tensor:
        """
        Get the transformed domain distribution parameters. The policy's parameters are in log space.

        :param params: policy parameters (can be the log of the actual domain distribution parameter value)
        :return: policy parameters transformed according to the mask
        """
        ddp = params.clone()
        ddp[self.mask] = to.exp(ddp[self.mask])
        return ddp

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is not None:
            # First check if there are some specific values to set
            self.param_values = init_values

        elif kwargs.get('prior', None) is not None:
            # Prior information is expected to be in form of a DomainRandomizer since it holds the distributions
            if not isinstance(kwargs['prior'], DomainRandomizer):
                raise pyrado.TypeErr(given=kwargs['prior'], expected_type=DomainRandomizer)

            # For every domain distribution parameter in the mapping, check if there is prior information
            for idx, ddp in self.mapping.items():
                for dp in kwargs['prior'].domain_params:
                    if ddp[0] == dp.name and ddp[1] in dp.get_field_names():
                        # The domain parameter exists in the prior and in the mapping
                        val = getattr(dp, f'{ddp[1]}')
                        if self.mask[idx]:
                            # Log-transform since it will later be exp-transformed
                            self.params[idx].fill_(to.log(to.tensor(val)))
                        else:
                            self.params[idx].fill_(to.tensor(val))

        else:
            # Last measure
            self.params.data.normal_(0, 1)
            print_cbt('Using uninformative random initialization for DomainDistrParamPolicy.', 'y')

    def clamp_params(self, params: to.Tensor) -> to.Tensor:
        """
        Project the policy parameters a.k.a. the domain distribution parameters to valid a range.

        :param params: parameter tensor with arbitrary values
        :return: parameters clipped to the bounds of the `EnvSpec` defined in the constructor
        """
        return clamp(params,
                     to.from_numpy(self.env_spec.act_space.bound_lo),
                     to.from_numpy(self.env_spec.act_space.bound_up))

    def forward(self, obs: to.Tensor = None) -> to.Tensor:
        # Should never be used. I know this might seem like an extreme abuse of the policy class but it is worth it.
        raise NotImplementedError
