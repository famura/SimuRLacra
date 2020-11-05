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
from typing import Sequence, Callable, Tuple

from pyrado.policies.feed_forward.fnn import FNN
from pyrado.policies.initialization import init_param
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import TwoHeadedPolicy


class TwoHeadedFNNPolicy(TwoHeadedPolicy):
    """ Policy architecture which has a common body and two heads that have a separate last layer """

    name: str = 'thfnn'

    def __init__(self,
                 spec: EnvSpec,
                 shared_hidden_sizes: Sequence[int],
                 shared_hidden_nonlin: [Callable, Sequence[Callable]],
                 head_1_size: int = None,
                 head_2_size: int = None,
                 head_1_output_nonlin: Callable = None,
                 head_2_output_nonlin: Callable = None,
                 shared_dropout: float = 0.,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param shared_hidden_sizes: sizes of shared hidden layer outputs. Every entry creates one shared hidden layer.
        :param shared_hidden_nonlin: nonlinearity for the shared hidden layers
        :param head_1_size: size of the fully connected layer for head 1, if `None` this is set to the action space dim
        :param head_2_size: size of the fully connected layer for head 2, if `None` this is set to the action space dim
        :param head_1_output_nonlin: nonlinearity for output layer of the first head
        :param head_2_output_nonlin: nonlinearity for output layer of the second head
        :param shared_dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        # Create the feed-forward neural network
        self.shared = FNN(
            input_size=spec.obs_space.flat_dim,
            output_size=shared_hidden_sizes[-1],
            hidden_sizes=shared_hidden_sizes,
            hidden_nonlin=shared_hidden_nonlin,
            dropout=shared_dropout,
            output_nonlin=None
        )

        # Create output layer
        head_1_size = spec.act_space.flat_dim if head_1_size is None else head_1_size
        head_2_size = spec.act_space.flat_dim if head_2_size is None else head_2_size
        self.head_1 = nn.Linear(shared_hidden_sizes[-1], head_1_size)
        self.head_2 = nn.Linear(shared_hidden_sizes[-1], head_2_size)
        self.head_1_output_nonlin = head_1_output_nonlin
        self.head_2_output_nonlin = head_2_output_nonlin

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            self.shared.init_param(None, **kwargs)
            init_param(self.head_1, **kwargs)
            init_param(self.head_2, **kwargs)
        else:
            self.param_values = init_values

    def forward(self, obs: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        obs = obs.to(self.device)

        # Get the output of the last shared layer and pass this to the two headers separately
        x = self.shared(obs)
        output_1 = self.head_1(x)
        output_2 = self.head_2(x)
        if self.head_1_output_nonlin is not None:
            output_1 = self.head_1_output_nonlin(output_1)
        if self.head_2_output_nonlin is not None:
            output_2 = self.head_2_output_nonlin(output_2)
        return output_1, output_2
