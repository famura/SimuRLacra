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

import math
import torch as to
from collections import Callable
from torch.optim.optimizer import Optimizer

import pyrado


class GSS(Optimizer):
    """ Golden Section Search optimizer """

    def __init__(self, params, param_min: to.Tensor, param_max: to.Tensor):
        # assert all(group['params'].size() == 1 for group in params)  # only for scalar params
        if not isinstance(param_min, to.Tensor):
            raise pyrado.TypeErr(given=param_min, expected_type=to.Tensor)
        if not isinstance(param_max, to.Tensor):
            raise pyrado.TypeErr(given=param_max, expected_type=to.Tensor)
        if not param_min.shape == param_max.shape:
            raise pyrado.ShapeErr(given=param_min, expected_match=param_max)
        if not all(param_min < param_max):
            raise pyrado.ValueErr(given=param_min, l_constraint=param_max)

        defaults = dict(param_min=param_min, param_max=param_max)
        super().__init__(params, defaults)
        self.gr = (math.sqrt(5) + 1)/2

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['lb'] = group['param_min']
                state['ub'] = group['param_max']

    def step(self, closure: Callable):
        """
        Performs a single optimization step.

        :param closure: a closure that reevaluates the model and returns the loss
        :return: accumulated loss for all parameter groups after the parameter update
        """
        loss = to.tensor([0.])
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['lb'] = group['param_min']
                    state['ub'] = group['param_max']

                state['step'] += 1

                # Compute new bounds for evaluating
                cand_lb = state['ub'] - (state['ub'] - state['lb'])/self.gr
                cand_ub = state['lb'] + (state['ub'] - state['lb'])/self.gr

                # Set param to lower bound and evaluate
                p.data = cand_lb
                loss_lb = closure()

                # Set param to upper bound and evaluate
                p.data = cand_ub
                loss_ub = closure()

                if loss_lb < loss_ub:
                    state['ub'] = cand_ub
                else:
                    state['lb'] = cand_lb

                # Set param to average value
                p.data = (state['ub'] + state['lb'])/2.

                # Accumulate the loss
                loss += closure()
        return loss
