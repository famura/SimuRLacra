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

import inspect
from typing import Callable, List, Optional, Sequence

import torch as to
import torch.nn as nn
from torch.jit import ScriptModule, export, script

from pyrado.policies.base import Policy
from pyrado.utils.data_types import EnvSpec


class TimePolicy(Policy):
    """A purely time-based policy, mainly useful for testing"""

    name: str = "time"

    def __init__(
        self, spec: EnvSpec, fcn_of_time: Callable[[float], Sequence[float]], dt: float, use_cuda: bool = False
    ):
        """
        Constructor

        :usage:
        .. code-block:: python

            policy = TimePolicy(env, lambda t: to.tensor([-sin(t) * 0.001]), 0.01)

        :param spec: environment specification
        :param fcn_of_time: time-depended function returning actions
        :param dt: time step [s]
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        # Script the function eagerly
        self._fcn_of_time = fcn_of_time
        self._dt = dt
        self._curr_time = None

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        self._curr_time = 0.0

    def forward(self, obs: Optional[to.Tensor] = None) -> to.Tensor:
        act = to.tensor(self._fcn_of_time(self._curr_time), dtype=to.get_default_dtype(), device=self.device)
        self._curr_time += self._dt
        return to.atleast_1d(act)

    def script(self) -> ScriptModule:
        return script(TraceableTimePolicy(self.env_spec, self._fcn_of_time, self._dt))


class TraceableTimePolicy(nn.Module):
    """
    A scriptable version of `TimePolicy`.

    We could try to make `TimePolicy` itself scriptable, but that won't work anyways due to Policy not being scriptable.
    Better to just write another class.
    """

    name: str = "time"

    # Attributes
    input_size: int
    output_size: int
    dt: float
    curr_time: float

    def __init__(self, spec: EnvSpec, fcn_of_time: Callable[[float], Sequence[float]], dt: float):
        """
        Constructor

        :param spec: environment specification
        :param fcn_of_time: time-depended function returning actions
        :param dt: time step [s]
        """
        super().__init__()
        self.env_spec = spec

        # Setup attributes
        self.input_size = spec.obs_space.flat_dim
        self.output_size = spec.act_space.flat_dim
        self.dt = dt
        self.curr_time = 0.0

        # Validate function signature
        sig = inspect.signature(fcn_of_time, follow_wrapped=False)
        posp = [
            p
            for p in sig.parameters.values()
            if p.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        assert len(posp) == 1
        param = next(iter(posp))
        # Check parameter type
        if param.annotation != float:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - parameter must have type float")
        # Check return type
        if sig.return_annotation != inspect.Signature.empty and sig.return_annotation != List[float]:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - return type must be List[float]")
        self.fcn_of_time = fcn_of_time

    @export
    def reset(self):
        """Reset the policy's internal state."""
        self.curr_time = 0.0

    def forward(self, obs: Optional[to.Tensor] = None) -> to.Tensor:
        act = to.tensor(self.fcn_of_time(self.curr_time), dtype=to.double)
        self.curr_time = self.curr_time + self.dt
        return act
