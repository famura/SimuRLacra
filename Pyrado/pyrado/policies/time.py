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
import torch as to
from typing import Callable, List, Sequence
from torch.jit import ScriptModule, script, export
from torch.nn import Module

from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy


class TimePolicy(Policy):
    """ A purely time-based policy, mainly useful for testing """

    name: str = 'time'

    def __init__(self, spec: EnvSpec, fcn_of_time: Callable[[float], Sequence[float]], dt: float, use_cuda: bool = False):
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

    def reset(self):
        self._curr_time = 0

    def forward(self, obs: to.Tensor) -> to.Tensor:
        act = to.tensor(self._fcn_of_time(self._curr_time), dtype=to.get_default_dtype())
        self._curr_time += self._dt
        return act

    def script(self) -> ScriptModule:
        return script(TraceableTimePolicy(self.env_spec, self._fcn_of_time, self._dt))


class TraceableTimePolicy(Module):
    """
    A scriptable version of TimePolicy.

    We could try to make TimePolicy itself scriptable, but that won't work anyways due to Policy not being scriptable.
    Better to just write another class.
    """

    # Attributes
    input_size: int
    output_size: int
    dt: float
    current_time: float

    def __init__(self, spec: EnvSpec, fcn_of_time: Callable[[float], Sequence[float]], dt: float):
        super().__init__()

        # Setup attributes
        self.input_size = spec.obs_space.flat_dim
        self.output_size = spec.act_space.flat_dim
        self.dt = dt

        self.env_spec = spec

        # Validate function signature
        sig = inspect.signature(fcn_of_time, follow_wrapped=False)
        posp = [p for p in sig.parameters.values() if p.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }]
        assert len(posp) == 1
        param = next(iter(posp))
        # check parameter type
        if param.annotation != float:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - parameter must have type float")
        # check return type
        if sig.return_annotation != inspect.Signature.empty and sig.return_annotation != List[float]:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - return type must be List[float]")

        self.fcn_of_time = fcn_of_time

        # setup current time buffer
        self.current_time = 0.

    @export
    def reset(self):
        self.current_time = 0.

    def forward(self, obs_ignore):
        act = to.tensor(self.fcn_of_time(self.current_time), dtype=to.double)
        self.current_time = self.current_time + self.dt
        return act
