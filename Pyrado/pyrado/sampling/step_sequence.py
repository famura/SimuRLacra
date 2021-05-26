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

import functools
import math
import operator
import random
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import scipy.signal as signal
import torch as to

import pyrado
from pyrado.sampling.data_format import cat_to_format, new_tuple, stack_to_format, to_format
from pyrado.sampling.utils import gen_ordered_batch_idcs, gen_shuffled_batch_idcs
from pyrado.utils.checks import check_all_equal, is_iterable
from pyrado.utils.data_types import EnvSpec


def _index_to_int(idx, n):
    # Index conversion
    idx = operator.index(idx)
    # Check negative index
    if idx < 0:
        idx += n
    # Check bounds
    if idx < 0 or idx >= n:
        raise IndexError
    return idx


class DictIndexProxy:
    """Views a slice through a dict of lists or tensors."""

    __slots__ = ("__dict__", "_obj", "_index", "_prefix")

    def __init__(self, obj: dict, index: int, path: Optional[str] = None):
        super().__init__()

        self._obj = obj
        self._index = index
        if path:
            self._prefix = path + "."
        else:
            self._prefix = ""

    def _process_key(self, key: str, index: int, error_type: Type[Exception]):
        return key, index

    def _get_keyed_value(self, key, error_type: Type[Exception] = RuntimeError):
        # Obtain keyed value from obj dict
        value = self._obj.get(key, None)

        if value is None:
            # Try pluralized keys
            value = self._obj.get(key + "s", None)
            if value is None:
                raise error_type(f"No entry named {self._prefix}{key}")

        return value

    def _index_value(self, key, value, index, error_type: Type[Exception] = RuntimeError):
        # Obtain indexed element from value
        if isinstance(value, dict):
            # Return subdict proxy
            return DictIndexProxy(value, index, self._prefix + key)
        elif isinstance(value, tuple):
            # Return tuple of slices
            # Since we can't proxy a tuple, we slice eagerly
            # Use type(value) to support named tuples. (the keys is still index though)
            return new_tuple(
                type(value), (self._index_value(f"{key}[{i}]", v, index, error_type) for i, v in enumerate(value))
            )
        elif isinstance(value, (to.Tensor, np.ndarray)):
            # Return slice of ndarray / tensor
            return value[index, ...]
        elif isinstance(value, list):
            # Return list item
            return value[index]
        else:
            # Unsupported type
            raise error_type(f"Entry {self._prefix}{key} has un-gettable type {type(value)}")

    def _get_indexed_value(self, key, error_type: Type[Exception] = RuntimeError):
        real_key, index = self._process_key(key, self._index, error_type)

        # Obtain keyed value list from obj dict
        value = self._get_keyed_value(real_key, error_type=error_type)

        return self._index_value(key, value, index, error_type)

    def _set_indexed_value(self, key, new_value, error_type: Type[Exception] = RuntimeError):
        real_key, index = self._process_key(key, self._index, error_type)

        # Obtain keyed value list from obj dict
        value = self._get_keyed_value(real_key, error_type=error_type)

        # Set value to data
        if isinstance(value, (to.Tensor, np.ndarray)):
            # Set slice of ndarray/tensor
            value[index, ...] = new_value
        elif isinstance(value, list):
            # Set list item
            value[index] = new_value
        else:
            # Don't support setting dict proxies
            raise error_type(f"Entry {key} has un-settable type {type(value)}")

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError
        result = self._get_indexed_value(key, error_type=AttributeError)
        self.__dict__[key] = result
        return result

    def __setattr__(self, key, value):
        if not key.startswith("_"):
            try:
                self._set_indexed_value(key, value, error_type=AttributeError)
            except AttributeError:
                pass
            else:
                self.__dict__[key] = value
                return
        object.__setattr__(self, key, value)

    def __dir__(self):
        # List dict items not starting with _
        return [k for k in self._obj if not k.startswith("_")]

    # Define getitem and setitem too, helps when return attr is a keyword
    def __getitem__(self, key):
        result = self._get_indexed_value(key, error_type=KeyError)
        self.__dict__[key] = result
        return result

    def __setitem__(self, key, value):
        self._set_indexed_value(key, value, error_type=KeyError)
        self.__dict__[key] = value

    # Serialize only dict and index
    def __getstate__(self):
        return {"obj", self._obj, "index", self._index}

    def __setstate__(self, state):
        self._obj = state["obj"]
        self._index = state["index"]


class Step(DictIndexProxy):
    """
    A single step in a rollout.

    This object is a proxy, referring a specific index in the rollout. When querying an attribute from the step,
    it will try to return the corresponding slice from the rollout. Additionally, one can prefix attributes with `next_`
    to access the value for the next step, i.e. `next_observations` the observation made at the start of the next step.
    """

    __slots__ = "_rollout"

    def __init__(self, rollout, index):
        """
        Constructor

        :param rollout: `StepSequence` object to which this step belongs
        :param index:  index of this step in the rollout
        """
        # Call DictIndexProxy's constructor
        super(Step, self).__init__(rollout.__dict__, index)

        self._rollout = rollout

    def _process_key(self, key: str, index: int, error_type: Type[Exception]):
        if key.startswith("next_"):
            if not self._rollout.continuous:
                raise error_type("Access to next element is not supported for non-continuous rollouts!")
            key = key[5:]
            index += 1
        if key not in self._rollout.data_names and key + "s" not in self._rollout.data_names and key != "done":
            raise error_type(f"No such rollout data field: {key}")
        return key, index

    # Serialize rollout and index
    def __getstate__(self):
        return {"rollout", self._rollout, "index", self._index}

    def __setstate__(self, state):
        self._rollout = state["rollout"]
        self._obj = self._rollout.__dict__
        self._index = state["index"]


class StepSequence(Sequence[Step]):
    """
    A sequence of steps.

    During the rollout, the values of different variables are recorded. This class provides efficient storage and
    access for these values. The constructor accepts a list of step entries for each variable. For every step,
    the list should contain a Tensor/ndarray of values for that step. The shape of these tensors must be the same for
    all step entries. The passed tensors are then stacked, so that the first dimension is the step count.
    Some values, like the observations, can have one more element then there are steps to encode the state after the
    last step. Additionally, the step entries may be dicts to support keyed storage. A list of dicts is converted to
    a dict of lists, each of which will be regularly stacked. Apart from the variable-based view, the rollout can also
    be seen as a sequence of steps. Each Step object is a proxy, it's attributes refer to the respective slice of the
    corresponding variable. The only required result variable are `rewards`, observations`, and `actions`.
    All other variables are optional. Common optional ones are `states` and `rollout_info`.

    .. note::
        Storing PyTorch tensors with gradient tracing is NOT supported. The rationale behind this is eager error
        avoidance. The only reason you would add them is to profit from the optimized slicing, but using that with
        gradient tracking risks lingering incomplete graphs.
    """

    rewards: Union[np.ndarray, to.Tensor]
    observations: Union[np.ndarray, to.Tensor]
    actions: Union[np.ndarray, to.Tensor]

    # Set of required rollout fields in addition to rewards, observations, actions. Putting this into a class field
    # instead of using the constructor arguments reduces duplicate code and allows to override it during unit tests.
    required_fields = {}

    def __init__(
        self,
        *,
        complete: bool = True,
        rollout_info=None,
        data_format: Optional[str] = None,
        done: Optional[np.ndarray] = None,
        continuous: bool = True,
        rollout_bounds=None,
        rewards: Sequence,
        observations: Sequence,
        actions: Sequence,
        **data,
    ):
        """
        Constructor

        :param complete: `False` if the rollout is incomplete, i.e. as part of a mini-batch
        :param rollout_info: data staying constant through the whole episode
        :param data_format: 'torch' to use Tensors, 'numpy' to use ndarrays.
                             Will use Tensors if any data argument does, else ndarrays
        :param done: boolean ndarray, specifying for each step whether it led to termination.
                     The last step of continuous rollouts, i.e. not mini-batches, is done if `complete` is `True`.
        :param continuous: true if the steps form one continuous sequence.
        :param rewards: sequence of reward values, determines sequence length
        :param observations: sequence of observation values, the length must be `len(rewards) + 1`
        :param actions: sequence of action values, the length must be `len(rewards)`
        :param data: additional data lists, their length must be `len(rewards)` or `len(rewards) + 1`
        """
        # Obtain rollout length from reward list
        self.length = len(rewards)
        if self.length == 0:
            raise pyrado.ShapeErr(msg="StepSequence cannot be empty!")

        # Set singular attributes
        self.rollout_info = rollout_info
        self.continuous = continuous

        # Infer if this instance is using numpy arrays or PyTorch tensors
        if data_format is None:
            # We ignore rewards here since it's probably scalar
            for value in data.values():
                if isinstance(value, to.Tensor) or (isinstance(value, list) and isinstance(value[0], to.Tensor)):
                    data_format = "torch"
                    break
            else:
                # Use numpy by default
                data_format = "numpy"
        self._data_format = data_format

        # Check for missing extra fields
        missing_fields = StepSequence.required_fields - data.keys()
        if missing_fields:
            raise ValueError(f"Missing required data fields: {missing_fields}")

        # Set mandatory data fields
        self._data_names = []
        self.add_data("rewards", rewards)
        self.add_data("observations", observations)
        self.add_data("actions", actions)

        # Set other data fields and verify their length
        for name, value in data.items():
            self.add_data(name, value)

        # Set done list if any. The done list is always a numpy array since torch doesn't support boolean tensors.
        if done is None:
            done = np.zeros(self.length, dtype=np.bool_)
            if complete and continuous:
                done[-1] = True
        else:
            done = np.asarray(done, dtype=np.bool_)
            assert done.shape[0] == self.length
        self.done = done

        # Compute rollout bounds from done list (yes this is not exactly safe...)
        # The bounds list has one extra entry 0, this simplifies queries greatly.
        # bounds[i] = start of rollout i; bounds[i+1]=end of rollout i
        if self.continuous:
            if rollout_bounds is None:
                rollout_bounds = [0]
                rollout_bounds.extend(np.flatnonzero(done) + 1)
                if not done[-1]:
                    rollout_bounds.append(self.length)
            else:
                # Validate externally passed bounds.
                for i in range(len(rollout_bounds) - 1):
                    assert rollout_bounds[i] < rollout_bounds[i + 1]
                assert rollout_bounds[0] == 0
                assert rollout_bounds[-1] == self.length
            self._rollout_bounds = np.array(rollout_bounds)
        else:
            self._rollout_bounds = None

    @property
    def data_format(self) -> str:
        """Get the name of data format ('torch' or 'numpy')."""
        return self._data_format

    @property
    def data_names(self) -> Sequence[str]:
        """Get the list of data attribute names."""
        return self._data_names

    @property
    def rollout_bounds(self) -> np.ndarray:
        assert self.continuous
        return self._rollout_bounds

    @property
    def rollout_count(self):
        """Count the number of sub-rollouts inside this step sequence."""
        if not self.continuous:
            raise pyrado.ValueErr(msg="Sub-rollouts are only supported on continuous data.")
        return len(self._rollout_bounds) - 1

    @property
    def rollout_lengths(self):
        """Lengths of sub-rollouts."""
        if not self.continuous:
            raise pyrado.ValueErr(msg="Sub-rollouts are only supported on continuous data.")
        bounds = self._rollout_bounds
        return bounds[1:] - bounds[:-1]

    def __len__(self):
        """Get the step sequence's length."""
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice) or isinstance(index, Iterable):
            # Return a StepSequence object with the subset. Build sliced data dict.
            sliced_data = {name: self._slice_entry(self.__dict__[name], index) for name in self._data_names}
            sliced_data = {k: v for k, v in sliced_data.items() if v is not None}

            # Check if the slice is continuous
            continuous = isinstance(index, slice) and (index.step is None or index.step == 1)
            rollout_bounds = None
            if continuous:
                # Slice rollout bounds too.
                start, end, _ = index.indices(self.length)
                rollout_bounds = [0]
                for b in self._rollout_bounds:
                    if start < b < end:
                        rollout_bounds.append(b - start)
                rollout_bounds.append(end - start)

            return StepSequence(
                rollout_info=self.rollout_info,
                data_format=self._data_format,
                done=self.done[index],
                continuous=continuous,
                rollout_bounds=rollout_bounds,
                **sliced_data,
            )

        # Should be a singular element index. Return step proxy.
        return Step(self, _index_to_int(index, self.length))

    def __map_tensors(self, mapper, elem):
        if isinstance(elem, dict):
            # Modify dict in-place
            for k in elem.keys():
                elem[k] = self.__map_tensors(mapper, elem[k])
            return elem
        if isinstance(elem, tuple):
            # Can't modify in place since it's a tuple
            return new_tuple(type(elem), (self.__map_tensors(mapper, part) for part in elem))

        # Tensor element
        return mapper(elem)

    def _validate_data_size(self, name, value):
        # In torch case: check that we don't mess with gradients
        if isinstance(value, to.Tensor):
            assert not value.requires_grad, (
                "Do not add gradient-sensitive tensors to SampleCollections. "
                "This is a fast road to weird retain_graph errors!"
            )

        # Check type of data
        if isinstance(value, dict):
            # Validate dict entries
            for k, v in value.items():
                self._validate_data_size(f"{name}.{k}", v)
            return
        if isinstance(value, tuple):
            # Validate dict entries
            for i, v in enumerate(value):
                self._validate_data_size(f"{name}[{i}]", v)
            return
        if isinstance(value, (np.ndarray, to.Tensor)):
            # A single array. The first dimension must match
            vlen = value.shape[0]
        else:
            # Should be a sequence
            assert isinstance(value, Sequence)
            vlen = len(value)

        if self.continuous:
            if not (vlen == self.length or vlen == self.length + 1):
                raise pyrado.ShapeErr(
                    msg=f"The data list {name} must have {self.length} or {self.length}+1 elements,"
                    f"but has {vlen} elements."
                )
        else:
            # Disallow +1 tensors
            if not vlen == self.length:
                raise pyrado.ShapeErr(
                    msg=f"The data list {name} must have {self.length} elements," f"but has {vlen} elements."
                )

    def _slice_entry(self, entry, index: slice):
        if isinstance(entry, dict):
            return {k: self._slice_entry(v, index) for k, v in entry.items()}
        if isinstance(entry, tuple):
            return new_tuple(type(entry), (self._slice_entry(e, index) for e in entry))
        elif isinstance(entry, (to.Tensor, np.ndarray)):
            return entry[index, ...]
        elif isinstance(entry, list):
            return entry[index]
        else:
            return None  # unsupported

    def _truncate_after_last(self, entry):
        if isinstance(entry, dict):
            return {k: self._truncate_after_last(v) for k, v in entry.items()}
        if isinstance(entry, tuple):
            return new_tuple(type(entry), (self._truncate_after_last(v) for v in entry))
        elif isinstance(entry, (to.Tensor, np.ndarray)):
            if entry.shape[0] == self.length + 1:
                return entry[:-1, ...]
        elif isinstance(entry, list):
            if len(entry) == self.length + 1:
                return entry[:-1]
        # No truncation
        return entry

    def add_data(self, name: str, value=None, item_shape: tuple = None, with_after_last: bool = False):
        """
        Add a new data field to the step sequence.

        :param name: string for the name
        :param value: the data
        :param item_shape: shape to store the data in
        :param with_after_last: `True` if there is one more element than the length (e.g. last observation)
        """
        if name in self._data_names:
            raise pyrado.KeyErr(msg=f"Trying to add a duplicate data field for {name}!")

        if value is None:
            # Compute desired step length
            ro_length = self.length
            if with_after_last:
                ro_length += 1

            # Create zero-filled
            if self._data_format == "torch":
                value = to.zeros(to.Size([ro_length]) + to.Size(item_shape))
            else:
                value = np.array((ro_length,) + item_shape)

        else:
            # Check the data
            self._validate_data_size(name, value)

            if not isinstance(value, (np.ndarray, to.Tensor)):
                # Stack into one array/tensor
                value = stack_to_format(value, self._data_format)
            else:
                # Ensure right array format
                value = to_format(value, self._data_format)

        # Store in dict
        self._data_names.append(name)
        self.__dict__[name] = value

    def get_data_values(self, name: str, truncate_last: bool = False):
        """
        Return the data tensor stored under the given name.

        :param name: data name
        :param truncate_last: True to truncate the length+1 entry if present
        """
        assert name in self._data_names
        entry = self.__dict__[name]

        # Truncate if needed
        if truncate_last:
            # Check length
            entry = self._truncate_after_last(entry)
        return entry

    def numpy(self, data_type=None):
        """
        Convert data to numpy `ndarray.

        :param data_type: type to return data in. When None is passed, the data type is left unchanged.
        """
        self.convert("numpy", data_type)

    def torch(self, data_type=None):
        """
        Convert data to PyTorch tensors.

        :param data_type: type to return data in. When None is passed, the data type is left unchanged.
        """
        self.convert("torch", data_type)

    def convert(self, data_format: str, data_type=None):
        """
        Convert data to specified format.

        :param data_format: torch to use Tensors, numpy to use ndarrays
        :param data_type: optional torch/numpy dtype for data. When `None` is passed, the data type is left unchanged.
        """
        if data_format not in {"torch", "numpy"}:
            raise pyrado.ValueErr(given=data_format, eq_constraint="'torch' or 'numpy'")

        if self._data_format == data_format:
            return
        self._data_format = data_format
        for dn in self._data_names:
            self.__dict__[dn] = self.__map_tensors(lambda t: to_format(t, data_format, data_type), self.__dict__[dn])

    def get_rollout(self, index):
        """
        Get an indexed sub-rollout.

        :param index: generic index of sub-rollout, negative values, slices and iterables are allowed
        :return: selected subset.
        """
        if not self.continuous:
            raise pyrado.ValueErr(msg="Sub-rollouts are only supported on continuous data.")
        if isinstance(index, slice):
            # Analyze slice
            start, end, step = index.indices(self.rollout_count)
            if step == 1:
                # A simple, continuous slice
                bounds = self._rollout_bounds
                start_step = bounds[start]
                end_step = bounds[end]
                return self[start_step:end_step]

            # Convert nonstandard slice to range
            index = range(start, end, step)
        if isinstance(index, Iterable):
            # Nontrivial non-continuous slice, need to slice each element and concat them.
            return StepSequence.concat([self.get_rollout(i) for i in index], self.data_format)

        # Decode index
        index = _index_to_int(index, self.rollout_count)
        bounds = self._rollout_bounds
        start_step = bounds[index]
        end_step = bounds[index + 1]
        return self[start_step:end_step]

    def iterate_rollouts(self):
        """Iterate over all sub-rollouts of a concatenated rollout."""
        if not self.continuous:
            raise pyrado.ValueErr(msg="Sub-rollouts are only supported on continuous data.")
        bounds = self._rollout_bounds
        count = len(bounds) - 1
        if count == 1:
            # Optimize for single rollout
            yield self
        else:
            for i in range(count):
                start_step = bounds[i]
                end_step = bounds[i + 1]
                yield self[start_step:end_step]

    def sample_w_next(self, batch_size: int) -> tuple:
        """
        Sample a random batch of steps from a together with the associated next steps.
        Similar to `split_shuffled_batches` with `complete_rollouts=False`

        :param batch_size: number of steps to sample
        :return: randomly sampled batch of steps
        """
        if not self.length >= 2:
            raise pyrado.ValueErr(given=self.length, ge_constraint="2")

        shuffled_idcs = random.sample(range(self.length - 2), batch_size)  # - 2 to always have a next step
        shuffled_next_idcs = [i + 1 for i in shuffled_idcs]
        steps = deepcopy(self[shuffled_idcs])
        next_steps = deepcopy(self[shuffled_next_idcs])

        return steps, next_steps

    def split_ordered_batches(self, batch_size: int = None, num_batches: int = None):
        """
        Batch generation. Split the step collection into ordered mini-batches of size batch_size.

        :param batch_size: number of steps per batch, i.e. variable number of batches
        :param num_batches: number of batches to split the rollout in, i.e. variable batch size

        .. note::
            Left out the option to return complete rollouts like for `split_shuffled_batches`.
        """
        if batch_size is None and num_batches is None or batch_size is not None and num_batches is not None:
            raise pyrado.ValueErr(msg="Either batch_size or num_batches must not be None, but not both or none!")
        elif batch_size is not None and batch_size < 1:
            raise pyrado.ValueErr(given=batch_size, ge_constraint="1 (int)")
        elif num_batches is not None and num_batches < 1:
            raise pyrado.ValueErr(given=num_batches, ge_constraint="1 (int)")

        # Switch the splitting mode
        if num_batches is not None:
            batch_size = math.ceil(self.length / num_batches)

        if batch_size >= self.length:
            # Yield all at once if there are less steps than the batch size
            yield self

        else:
            # Split by steps
            for b in gen_ordered_batch_idcs(batch_size, self.length, sorted=True):
                yield self[b]

    def split_shuffled_batches(self, batch_size: int, complete_rollouts: bool = False):
        """
        Batch generation. Split the step collection into random mini-batches of size batch_size.

        :param batch_size: number of steps per batch
        :param complete_rollouts: if `complete_rollouts = True`, the batches will not contain partial rollouts.
                                  However, the size of the returned batches cannot be strictly maintained in this case.

        .. note::
            This method is also supposed to be called for recurrent networks, which have a different `evaluate()`
            method that recognized where the rollouts end within a batch.
        """
        if batch_size >= self.length:
            # Yield all at once if there are less steps than the batch size
            yield self

        elif complete_rollouts and self.continuous:
            # Our goal here is to randomly shuffle the rollouts, while returning batches of batch_size steps.
            # The solution here is to take rollouts in a random order and yield a batch each time it exceeds batch_size.

            rollout_lengths = self.rollout_lengths
            shuffled_idcs = random.sample(range(len(rollout_lengths)), len(rollout_lengths))
            # Now, walk through the rollouts in a random order and split once batch size is full.
            batch = []
            cur_batch_size = 0
            for idx in shuffled_idcs:
                batch.append(idx)
                cur_batch_size += rollout_lengths[idx]
                if cur_batch_size >= batch_size:
                    # Got a full batch
                    yield self.get_rollout(batch)
                    batch.clear()
                    cur_batch_size = 0
            # Yield eventual final one
            if batch:
                yield self.get_rollout(batch)

        else:
            # Split by steps
            for b in gen_shuffled_batch_idcs(batch_size, self.length):
                yield self[b]

    def undiscounted_return(self) -> float:
        """
        Compute the undiscounted return.

        :return: sum of rewards
        """
        if not len(self._rollout_bounds) == 2:
            raise pyrado.ShapeErr(msg="The StepSequence must be a single continuous rollout.")

        return self.rewards.sum()

    def discounted_return(self, gamma: float) -> (to.Tensor, np.ndarray):
        """
        Compute the discounted return.

        :param gamma: temporal discount factor
        :return: exponentially weighted sum of rewards
        """
        if not len(self._rollout_bounds) == 2:
            raise pyrado.ShapeErr(msg="The StepSequence must be a single continuous rollout.")
        if not 0 <= gamma <= 1:
            raise pyrado.ValueErr(given=gamma, ge_constraint="0", le_constraint="1")

        if self.data_format == "torch":
            return to.dot(self.rewards, (gamma ** to.arange(self.length)))
        else:
            return np.dot(self.rewards, (gamma ** np.arange(self.length)))

    @classmethod
    def concat(cls, parts: Sequence["StepSequence"], data_format: Optional[str] = None, truncate_last: bool = True):
        """
        Concatenate multiple step sequences into one, truncating the last observation.

        :param parts: batch of sequences to concatenate
        :param data_format: torch to use Tensors, numpy to use ndarrays, `None` to choose automatically
        :param truncate_last: remove the last step from each part, highly recommended to be `True`
        :return: concatenated sequence of `Steps`
        """
        # Obtain data attribute names
        data_names = parts[0].data_names

        # Deduce data format if is None
        if data_format is None:
            data_format = parts[0].data_format

        # Concat data fields
        data = {
            name: cat_to_format([ro.get_data_values(name, truncate_last) for ro in parts], data_format)
            for name in data_names
        }

        # Treat done separately since it should stay a ndarray
        done = np.concatenate([ro.done for ro in parts])

        # Check if parts are continuous
        continuous = all(ro.continuous for ro in parts)
        rollout_bounds = None
        if continuous:
            # Concatenate rollout separator indices for continuous rollouts
            rollout_bounds = [0]
            acc_len = 0
            for ro in parts:
                rollout_bounds.extend(ro.rollout_bounds[1:] + acc_len)
                acc_len += ro.rollout_bounds[-1]

        # Keep the rollout info dict if it is the same for all StepSequences
        rollout_info = None
        if check_all_equal([p.rollout_info for p in parts]):
            rollout_info = parts[0].rollout_info

        return StepSequence(
            data_format=data_format,
            done=done,
            continuous=continuous,
            rollout_bounds=rollout_bounds,
            rollout_info=rollout_info,
            **data,
        )

    @classmethod
    def process_data(
        cls,
        rollout: "StepSequence",
        fcn: Callable,
        fcn_arg_name: str,
        fcn_arg_types: Union[type, Tuple[type]] = np.ndarray,
        include_fields: Sequence[str] = None,
        exclude_fields: Sequence[str] = None,
        **process_fcn_kwargs,
    ):
        """
        Process all data fields of a rollouts using an arbitrary function. Optionally, some fields can be excluded.

        :param rollout: `StepSequence` holding the data
        :param fcn: function (of one remaining input) to used manipulate the data fields, e.g. `scipy.filtfilt()`
        :param fcn_arg_name: sting of the remaining input of `process_fcn()`, e.g. `x` for `scipy.filtfilt()`
        :param fcn_arg_types: type or tuple thereof which are expected as input to `fcn()`
        :param include_fields: list of field names to include for processing, pass `None` to not include everything.
                               If specified, only fields from this selection will be considered
        :param exclude_fields: list of field names to exclude from processing, pass `None` to not exclude anything
        :param process_fcn_kwargs: keyword arguments forwarded to `process_fcn()`
        :return: new `StepSequence` instance with processed data
        """

        @functools.wraps(fcn)
        def recursive_wrapper(inp, **kwargs):
            """Wrap the processing function to call it recursivelyy for nested data structures."""
            # Add to actual data input to the keyword arguments to make calling the function easier
            kwargs.update({fcn_arg_name: inp})

            if isinstance(inp, fcn_arg_types):
                # Process the data
                inp = fcn(**kwargs)

            elif isinstance(inp, dict):
                # Recursive call
                for key, value in inp.items():
                    if isinstance(value, fcn_arg_types):
                        inp[key] = recursive_wrapper(value, **kwargs)
                    else:
                        inp[key] = value

            elif isinstance(inp, list):
                # Recursive call
                for idx, item in enumerate(inp):
                    if isinstance(item, fcn_arg_types):
                        inp[idx] = recursive_wrapper(item, **kwargs)
                    else:
                        inp[idx] = item

            return inp

        # Go through all desired data fields and apply the processing function
        data_dict = dict()
        include_fields = include_fields or rollout.data_names
        exclude_fields = exclude_fields or []
        for name in rollout.data_names:
            # Extract data field
            data = rollout.get_data_values(name)

            # Process current data field if included and not explicitly excluded
            if name in include_fields and name not in exclude_fields:
                data = recursive_wrapper(data, **process_fcn_kwargs)

            # Collect the new/old data
            data_dict[name] = data

        # Create new object
        return StepSequence(**data_dict, rollout_info=rollout.rollout_info, continuous=rollout.continuous)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, env_spec: EnvSpec, continuous: bool = True) -> "StepSequence":
        """
        Generate a StepSequence object from a Pandas DataFrame instance.
        Not all data fields are supported. The fields 'rewards' is mandatory.

        :param df: Pandas DataFrame holding the data in 1-dim arrays
        :param env_spec: environment specifications which labels are used to slice the DataFrame
        :param continuous: `True` if the rollout to be reconstructed was continuous
        :return: new StepSequence
        """
        if (
            env_spec.state_space.labels is None
            or env_spec.obs_space.labels is None
            or env_spec.act_space.labels is None
        ):
            raise pyrado.ValueErr(
                msg="The provided EnvSpec instance must contain labels for the state, observation, and action space "
                "since these are used to filter the pandas DataFrame by!"
            )

        # Mandatory fields
        rewards = df["rewards"].to_numpy()
        states = df[env_spec.state_space.labels].to_numpy()
        observations = df[env_spec.obs_space.labels].to_numpy()
        actions = df[env_spec.act_space.labels].to_numpy()

        # Remove NaNs which come from concatenating columns of different length in Pandas
        rewards = rewards[~np.isnan(rewards)]
        actions = actions[~np.isnan(actions).any(axis=1), :]  # check if any in a column is none for multi-dim case

        # Other fields
        mandatory_labels = (
            ["rewards"]
            + env_spec.state_space.labels.tolist()
            + env_spec.obs_space.labels.tolist()
            + env_spec.act_space.labels.tolist()
        )
        other = df.drop(columns=mandatory_labels)
        other_dict = dict()
        for name, data in other.iteritems():
            other_dict[name] = data.values

        return StepSequence(
            rewards=rewards,
            states=states,
            actions=actions,
            observations=observations,
            continuous=continuous,
            **other_dict,
        )


def discounted_reverse_cumsum(data, gamma: float):
    """
    Use a linear filter to compute the reverse discounted cumulative sum.

    .. note::
        `scipy.signal.lfilter` assumes an initialization with 0 by default.

    :param data: input data with samples along the 0 axis (e.g. time series)
    :param gamma: discount factor
    :return: cumulative sums for every step
    """
    return signal.lfilter([1], [1, -gamma], data[::-1], axis=0)[::-1]


def discounted_value(rollout: StepSequence, gamma: float):
    """
    Compute the discounted state values for one rollout.

    :param rollout: input data
    :param gamma: temporal discount factor
    :return: state values for every time step in the rollout
    """
    rewards = [step.reward for step in rollout]
    return discounted_reverse_cumsum(rewards, gamma)


def discounted_values(rollouts: Sequence[StepSequence], gamma: float, data_format: Optional[str] = "torch"):
    """
    Compute the discounted state values for multiple rollouts.

    :param rollouts: input data
    :param gamma: temporal discount factor
    :param data_format: data format of the given
    :return: state values for every time step in the rollouts (concatenated sequence across rollouts)
    """
    if data_format == "torch":
        # The ndarray.copy() is necessary due to (currently) unsupported negative strides
        return to.cat([to.from_numpy(discounted_value(ro, gamma).copy()).to(to.get_default_dtype()) for ro in rollouts])
    elif data_format == "numpy":
        return np.array([discounted_value(ro, gamma) for ro in rollouts])
    else:
        raise pyrado.ValueErr(given=data_format, eq_constraint="'torch' or 'numpy'")


def gae_returns(rollout: StepSequence, gamma: float = 0.99, lamb: float = 0.95):
    """
    Compute returns using generalized advantage estimation.

    .. seealso::
        [1] J. Schulmann, P. Moritz, S. Levine, M. Jordan, P. Abbeel, 'High-Dimensional Continuous Control Using
        Generalized Advantage Estimation', ICLR 2016

    :param rollout: sequence of steps
    :param gamma: temporal discount factor
    :param lamb: discount factor
    :return: estimated advantage
    """

    def _next_value(step: Step) -> float:
        """Helper to return `next_value = 0` for last step"""
        if step.done:
            return 0.0
        return step.next_value

    deltas = [step.reward + gamma * _next_value(step) - step.value for step in rollout]
    cumsum = discounted_reverse_cumsum(deltas, gamma * lamb)
    return cumsum


def check_act_equal(
    rollout_1: Union[StepSequence, List[StepSequence]],
    rollout_2: Union[StepSequence, List[StepSequence]],
    check_applied: bool = False,
):
    """
    Check if the actions of two rollouts or pairwise two rollouts in in two lists are approximately the same

    :param rollout_1: rollouts or list of rollouts
    :param rollout_2: rollouts or list of rollouts
    :param check_applied: if `True` check the actions applied to the environment instead of the commanded ones
    :return: `True` if the actions match
    """
    act_key = "actions_applied" if check_applied else "actions"

    if isinstance(rollout_1, StepSequence) and isinstance(rollout_2, StepSequence):
        if not np.allclose(
            rollout_1.get_data_values(act_key)[: min(rollout_1.length, rollout_2.length)],
            rollout_2.get_data_values(act_key)[: min(rollout_1.length, rollout_2.length)],
        ):
            raise pyrado.ValueErr(msg="The actions in the rollouts to compare are not equal!")

    elif is_iterable(rollout_1) and is_iterable(rollout_2):
        if not all(
            [
                np.allclose(
                    r1.get_data_values(act_key)[: min(r1.length, r2.length)],
                    r2.get_data_values(act_key)[: min(r1.length, r2.length)],
                )
                for r1, r2 in zip(rollout_1, rollout_2)
            ]
        ):
            raise pyrado.ValueErr(msg="The actions in the rollouts to compare are not equal!")

    else:
        raise NotImplementedError
