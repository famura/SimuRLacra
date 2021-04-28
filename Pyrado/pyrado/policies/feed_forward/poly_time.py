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

from typing import List, Optional, Union

import torch as to
import torch.nn as nn

import pyrado
from pyrado.policies.base import Policy
from pyrado.utils.data_processing import correct_atleast_2d
from pyrado.utils.data_types import EnvSpec


class PolySplineTimePolicy(Policy):
    """A purely time-based policy, were the output is given by a pol"""

    name: str = "pst"

    def __init__(
        self,
        spec: EnvSpec,
        dt: float,
        t_end: float,
        cond_lvl: str,
        cond_final: Union[to.Tensor, List[float]],
        cond_init: Optional[Union[to.Tensor, List[float]]] = None,
        t_init: float = 0.0,
        init_param_kwargs: Optional[dict] = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param dt: time step [s]
        :param t_end: final time [s], relative to `t_init`
        :param cond_lvl: highest level of the condition, so far, only velocity 'vel' and acceleration 'acc' level
                         conditions on the polynomial are supported. These need to be consistent with the actions.
        :param cond_final: final condition for the least squares proble,, needs to be of shape [X, dim_act] where X is
                           2 if `cond_lvl == 'vel'` and 4 if `cond_lvl == 'acc'`
        :param cond_init: initial condition for the least squares proble,, needs to be of shape [X, dim_act] where X is
                           2 if `cond_lvl == 'vel'` and 4 if `cond_lvl == 'acc'`
        :param t_init: initial time [s], also used on calling `reset()`, relative to `t_end`
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if t_end <= t_init:
            raise pyrado.ValueErr(given=t_end, g_constraint=t_init)

        # Call Policy's constructor
        super().__init__(spec, use_cuda)

        self._dt = dt
        self._t_end = t_end
        self._t_init = t_init
        self._curr_time = t_init

        # Determine the initial and final conditions used to compute the coefficients of the polynomials
        if cond_lvl.lower() == "vel":
            self._order = 3
        elif cond_lvl.lower() == "acc":
            self._order = 5
        else:
            raise pyrado.ValueErr(given=cond_lvl, eq_constraint="'vel' or 'acc'")
        num_cond = (self._order + 1) // 2
        cond_final = to.as_tensor(cond_final, dtype=to.get_default_dtype())
        cond_final = correct_atleast_2d(to.atleast_2d(cond_final))
        if cond_final.shape != (num_cond, spec.act_space.flat_dim):
            raise pyrado.ShapeErr(given=cond_final, expected_match=(num_cond, spec.act_space.flat_dim))
        if cond_init is not None:
            cond_init = to.as_tensor(cond_init, dtype=to.get_default_dtype())
            cond_init = correct_atleast_2d(to.atleast_2d(cond_init))
            if cond_init.shape != (num_cond, spec.act_space.flat_dim):
                raise pyrado.ShapeErr(given=cond_init, expected_match=(num_cond, spec.act_space.flat_dim))
        else:
            cond_init = to.zeros(num_cond, spec.act_space.flat_dim)
        self.conds = to.cat([cond_init, cond_final], dim=0)
        assert self.conds.shape[0] in [4, 6]

        # Store the polynomial coefficients for each output dimension in a matrix
        self.net = nn.Linear(self._order + 1, spec.act_space.flat_dim, bias=False)

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    @to.no_grad()
    def _compute_coefficients(self):
        """
        Compute the coefficients of the polynomial spline, and set them into the internal linear layer for storing.
        """
        # Treat each action dimension separately
        for idx_act in range(self.env_spec.act_space.flat_dim):
            # Get the feature matrices for both points in time
            feats = to.cat([self._compute_feats(self._t_init), self._compute_feats(self._t_end)], dim=0)

            # Solve least squares problem
            coeffs = to.lstsq(self.conds[:, idx_act], feats).solution

            # Store
            self.net.weight[idx_act, :] = coeffs.squeeze()

    @to.no_grad()
    def _compute_feats(self, t: float) -> to.Tensor:
        """
        Compute the feature matrix depending on the time and the number of conditions.

        :param t: time to evaluate at [s]
        :return: feature matrix, either of shape [2, 4], or shape [3, 6]
        """
        if self.conds.shape[0] == 4:
            # 3rd order polynomials, i.e. position and velocity level constrains
            feats = to.tensor(
                [
                    [1.0, t, t ** 2, t ** 3],
                    [0.0, 1.0, 2 * t, 3 * t ** 2],
                ]
            )
        else:
            # 5th order polynomials, i.e. position velocity, and acceleration level constrains
            feats = to.tensor(
                [
                    [1.0, t, t ** 2, t ** 3, t ** 4, t ** 5],
                    [0.0, 1.0, 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4],
                    [0.0, 0.0, 2.0, 6 * t, 12 * t ** 2, 20 * t ** 3],
                ]
            )
        return feats

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            self._compute_coefficients()
        else:
            self.param_values = init_values  # ignore the IntelliJ warning

    def reset(self):
        self._curr_time = self._t_init

    def forward(self, obs: Optional[to.Tensor] = None) -> to.Tensor:  # pylint: disable=arguments-differ,unused-argument
        # Get a vector of powers of the current time
        t_powers = to.tensor([self._curr_time ** o for o in range(self._order + 1)], dtype=to.get_default_dtype())

        act = self.net(t_powers)  # matrix-vector multiplication
        self._curr_time += self._dt

        return to.atleast_1d(act)
