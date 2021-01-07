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
from nflows.flows import Flow
from typing import Optional, Union, Dict

import pyrado
from pyrado.spaces import BoxSpace
from pyrado.spaces.empty import EmptySpace
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy
from pyrado.policies.initialization import init_param


class NFlowPolicy(Policy):
    """ Feed-forward policy powered by a normalizing flow """

    name: str = "flow"

    def __init__(
        self,
        flow: Flow,
        mapping: Dict[int, str],
        trafo_mask: Union[list, to.Tensor],
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param flow: normalizing flow from nflows module
        :param mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass, length).
                        The integers are indices of the numpy array which come from the algorithm.
        :param trafo_mask: every domain parameter that is set to `True` in this mask will be learned via a 'virtual'
                           parameter, i.e. in sqrt-space, and then finally squared to retrieve the domain parameter.
                           This transformation is useful to avoid setting a negative variance.
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(mapping, dict):
            raise pyrado.TypeErr(given=mapping, expected_type=dict)
        if not len(trafo_mask) == len(mapping):
            raise pyrado.ShapeErr(given=trafo_mask, expected_match=mapping)
        if not isinstance(flow, Flow):
            raise pyrado.TypeErr(given=flow, expected_type=Flow)

        # Define the parameter space by using the Policy.env_spec.act_space
        param_spec = EnvSpec(
            obs_space=EmptySpace(),  # TODO this could be used for the context
            act_space=BoxSpace(-pyrado.inf, pyrado.inf, shape=len(mapping)),
        )

        # Call Policy's constructor
        super().__init__(param_spec, use_cuda)

        self._mapping = mapping
        self.mask = to.tensor(trafo_mask, dtype=to.bool)
        self.flow = flow

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)

    @property
    def mapping(self) -> Dict[int, str]:
        """ Get the mapping from subsequent integers (starting at 0) to domain parameter names. """
        return self._mapping

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize the normalizing flow using default initialization
            init_param(self.flow, **kwargs)
        else:
            self.param_values = init_values

    def forward(self, num_samples: int, context: Optional[to.Tensor] = None, batch_size: Optional[int] = None):
        # Sample untransformed domain parameters using the normalizing flow
        dps_raw = self.sample(num_samples, context, batch_size)

        # Transform the domain parameters of
        return self.transform_to_dp_space(dps_raw)

    def sample(
        self, num_samples: int, context: Optional[to.Tensor] = None, batch_size: Optional[int] = None
    ) -> to.Tensor:
        """
        Forward to the sample function from `nflows` to generate samples from the distribution.

        :param num_samples: number of samples to generate
        :param context: conditioning variables. If `None`, the context is ignored.
        :param batch_size: number of samples per batch. If `None`, all samples are generated in one batch.
        :return: tensor containing the samples, with shape [num_samples, ...] if context is `None`, or
                 [context_size, num_samples, ...] if a context is given.
        """
        return self.flow.sample(num_samples, context, batch_size)

    def log_prob(self, inputs, context: Optional[to.Tensor] = None):
        """
        Forward to the log probability calculation from `nflows`.

        :param inputs: input variables
        :param context: conditioning variables. If it is a tensor, it must have the same number or rows as the inputs.
                        If `None`, the context is ignored.
        :return: tensor of shape [input_size], the log probability of the inputs given the context
        """
        return self.flow.log_prob(inputs, context)

    def transform_to_dp_space(self, params: to.Tensor) -> to.Tensor:
        """
        Get the transformed domain parameters. Where ever the mask is `True`, the corresponding domain parameter is
        learned in log space.

        :param params: raw domain parameters, i.e. the output of the normalizing flow
        :return: domain parameters transformed according to the mask
        """
        domain_params = params.clone()

        if domain_params.ndimension() == 1:
            # Only one set of domain parameters
            domain_params[self.mask] = to.pow(domain_params[self.mask], 2)

        elif domain_params.ndimension() == 2:
            # Multiple sets of domain parameters
            domain_params[:, self.mask] = to.pow(domain_params[:, self.mask], 2)

        elif domain_params.ndimension() == 3:
            # Multiple sets of domain parameters  with context
            domain_params[:, :, self.mask] = to.pow(domain_params[:, :, self.mask], 2)

        else:
            raise pyrado.ShapeErr(msg="The input must not have more than 3 dimensions!")

        return domain_params
