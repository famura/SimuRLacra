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
import numpy as np
import torch as to
from torch.jit import ScriptModule, script, export
from torch.nn import Module
from typing import Callable, List, Sequence, Optional, Union

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy


class TimePolicy(Policy):
    """ A purely time-based policy, mainly useful for testing """

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

    def reset(self):
        self._curr_time = 0.0

    def forward(self, obs: Optional[to.Tensor] = None) -> to.Tensor:
        act = to.tensor(self._fcn_of_time(self._curr_time), dtype=to.get_default_dtype())
        self._curr_time += self._dt
        return to.atleast_1d(act)

    def script(self) -> ScriptModule:
        return script(TraceableTimePolicy(self.env_spec, self._fcn_of_time, self._dt))


class TraceableTimePolicy(Module):
    """
    A scriptable version of TimePolicy.

    We could try to make TimePolicy itself scriptable, but that won't work anyways due to Policy not being scriptable.
    Better to just write another class.
    """

    name: str = "trtime"

    # Attributes
    input_size: int
    output_size: int
    dt: float
    current_time: float

    def __init__(self, spec: EnvSpec, fcn_of_time: Callable[[float], Sequence[float]], dt: float):
        super().__init__()
        self.env_spec = spec

        # Setup attributes
        self.input_size = spec.obs_space.flat_dim
        self.output_size = spec.act_space.flat_dim
        self.dt = dt
        self.current_time = 0.0

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
        # check parameter type
        if param.annotation != float:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - parameter must have type float")
        # check return type
        if sig.return_annotation != inspect.Signature.empty and sig.return_annotation != List[float]:
            raise ValueError(f"Malformed fcn_of_time with signature {sig} - return type must be List[float]")
        self.fcn_of_time = fcn_of_time

    @export
    def reset(self):
        self.current_time = 0.0

    def forward(self, obs: Optional[to.Tensor] = None) -> to.Tensor:
        act = to.tensor(self.fcn_of_time(self.current_time), dtype=to.double)
        self.current_time = self.current_time + self.dt
        return act


class PlaybackPolicy(Policy):
    """ A policy wish simply replays a sequence of actions. If more actions are requested, the policy  """

    name: str = "pb"

    def __init__(
        self,
        spec: EnvSpec,
        act_recordings: List[Union[to.Tensor, np.array]],
        no_reset: bool = False,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param act_recordings: pre-recorded sequence of actions to be played back later
        :param no_reset: `True` to turn `reset()` into a dummy function
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(act_recordings, list):
            raise pyrado.TypeErr(given=act_recordings, expected_type=list)

        super().__init__(spec, use_cuda)

        self._curr_rec = -1  # is increased before the first use
        self._curr_step = 0
        self._no_reset = no_reset
        self._num_rec = len(act_recordings)
        self._act_rec_buffer = [to.atleast_2d(to.as_tensor(ar)) for ar in act_recordings]
        if not all(b.shape[1] == self.env_spec.act_space.flat_dim for b in self._act_rec_buffer):
            raise pyrado.ShapeErr(
                given=(-1, self._act_rec_buffer[0].shape[1]), expected_match=(-1, self.env_spec.act_space.flat_dim)
            )

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    @property
    def curr_step(self) -> int:
        """ Get the number of the current replay step (0 for the initial step). """
        return self._curr_step

    @curr_step.setter
    def curr_step(self, curr_step: int):
        """ Set the number of the current replay step (0 for the initial step). """
        if not isinstance(curr_step, int) or not 0 <= curr_step < len(self._act_rec_buffer[self._curr_rec]):
            raise pyrado.ValueErr(
                given=curr_step, ge_constraint="0 (int)", l_constraint=len(self._act_rec_buffer[self._curr_rec])
            )
        self._curr_step = curr_step

    @property
    def curr_rec(self) -> int:
        """ Get the pointer to the current recording. """
        return self._curr_rec

    @curr_rec.setter
    def curr_rec(self, curr_rec: int):
        """ Set the pointer to the current recording. """
        if not isinstance(curr_rec, int) or not -1 <= curr_rec < len(self._act_rec_buffer):
            raise pyrado.ValueErr(given=curr_rec, ge_constraint="-1 (int)", l_constraint=len(self._act_rec_buffer))
        self._curr_rec = curr_rec

    def reset_curr_rec(self):
        """ Reset the pointer to the current recording. """
        self._curr_rec = -1

    def reset(self):
        if not self._no_reset:
            # Start at the beginning of the next recording
            self._curr_rec = (self._curr_rec + 1) % self._num_rec
            self._curr_step = 0

    def forward(self, obs: Optional[to.Tensor] = None) -> to.Tensor:
        if self._curr_step < len(self._act_rec_buffer[self._curr_rec]):
            # Asking for something that is available, return the stored action
            act = self._act_rec_buffer[self._curr_rec][self._curr_step, :]
        else:
            # Asking for more actions than provided, return zeros
            act = to.zeros(self.env_spec.act_space.shape)
        self._curr_step += 1
        return act

    def script(self) -> ScriptModule:
        raise NotImplementedError
