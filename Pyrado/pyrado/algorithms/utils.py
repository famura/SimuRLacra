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
import numpy as np
import torch as to
from copy import deepcopy
from torch.distributions import Distribution
from typing import NamedTuple, Union, Sequence, Callable

import pyrado
from pyrado.sampling.step_sequence import StepSequence
from pyrado.exploration.stochastic_action import StochasticActionExplStrat
from pyrado.utils.input_output import print_cbt


class ActionStatistics(NamedTuple):
    r"""
    act_distr: probability distribution at the given policy output values
    log_probs: $\log (p(act|obs, hidden))$ if hidden exists, else $\log (p(act|obs))$
    entropy: entropy of the action distribution
    """
    act_distr: Distribution
    log_probs: to.Tensor
    entropy: to.Tensor


def compute_action_statistics(steps: StepSequence, expl_strat: StochasticActionExplStrat) -> ActionStatistics:
    r"""
    Get the action distribution from the exploration strategy, compute the log action probabilities and entropy
    for the given rollout using the given exploration strategy.

    .. note::
        Requires the exploration strategy to have a (most likely custom) `evaluate()` method.

    :param steps: recorded rollout data
    :param expl_strat: exploration strategy used to generate the data
    :return: collected action statistics, see `ActionStatistics`
    """
    # Evaluate rollout(s)
    distr = expl_strat.evaluate(steps)

    # Collect results
    return ActionStatistics(distr, distr.log_prob(steps.actions.to(expl_strat.device)), distr.entropy())


class ReplayMemory:
    """ Base class for storing step transitions """

    def __init__(self, capacity: int):
        """
        Constructor

        :param capacity: number of steps a.k.a. transitions in the memory
        """
        self.capacity = int(capacity)
        self._memory = None

    @property
    def memory(self) -> StepSequence:
        """ Get the replay buffer. """
        return self._memory

    @property
    def isempty(self) -> bool:
        """ Check if the replay buffer is empty. """
        return self._memory is None

    def __len__(self) -> int:
        """ Get the number of transitions stored in the buffer. """
        return self._memory.length

    def push(self, ros: Union[list, StepSequence], truncate_last: bool = True):
        """
        Save a sequence of steps and drop of steps if the capacity is exceeded.

        :param ros: list of rollouts or one concatenated rollout
        :param truncate_last: remove the last step from each rollout, forwarded to `StepSequence.concat`
        """
        if isinstance(ros, list):
            # Concatenate given rollouts if necessary
            ros = StepSequence.concat(ros)
        elif isinstance(ros, StepSequence):
            pass
        else:
            pyrado.TypeErr(given=ros, expected_type=[list, StepSequence])

        # Add new steps
        if self.isempty:
            self._memory = deepcopy(ros)  # on the very first call
        else:
            self._memory = StepSequence.concat([self._memory, ros], truncate_last=truncate_last)

        num_surplus = self._memory.length - self.capacity
        if num_surplus > 0:
            # Drop surplus of old steps
            self._memory = self._memory[num_surplus:]

    def sample(self, batch_size: int) -> tuple:
        """
        Sample randomly from the replay memory.

        :param batch_size: number of samples
        :return: tuple of transition steps and associated next steps
        """
        return self._memory.sample_w_next(batch_size)

    def reset(self):
        self._memory = None

    def avg_reward(self) -> float:
        """
        Compute the average reward for all steps stored in the replay memory.

        :return: average reward
        """
        if self._memory is None:
            raise pyrado.TypeErr(msg='The replay memory is empty!')
        else:
            return sum(self._memory.rewards)/self._memory.length


def until_thold_exceeded(thold: float, max_iter: int = None):
    """
    Designed to wrap a function and repeat it until the return value exceeds a threshold.

    :param thold: threshold
    :param max_iter: maximum number of iterations of the wrapped function, set to `None` to run the loop relentlessly
    :return: wrapped function
    """

    def actual_decorator(trn_eval_fcn):
        """
        Designed to wrap a training + evaluation function and repeat it  it until the return value exceeds a threshold.

        :param trn_eval_fcn: function to wrap
        :return: wrapped function
        """

        @functools.wraps(trn_eval_fcn)
        def wrapper_trn_eval_fcn(*args, **kwargs):
            ret = -pyrado.inf
            cnt_iter = 0
            while ret <= thold:  # <= guarantees that we at least train once, even if thold is -inf
                # Train and evaluate
                ret = trn_eval_fcn(*args, **kwargs)
                cnt_iter += 1

                # Break if done
                if ret >= thold:
                    print_cbt(f'The policy exceeded the threshold {thold}.', 'g', True)
                    break

                # Break if max_iter is reached
                if max_iter is not None and cnt_iter == max_iter:
                    print_cbt(f'Exiting the training and evaluation loop after {max_iter} iterations.', 'y', True)
                    break

                # Else repeat training
                print_cbt(f'The policy did not exceed the threshold {thold}. Repeating training and evaluation ...',
                          'w', True)
            return ret

        return wrapper_trn_eval_fcn

    return actual_decorator


def num_iter_from_rollouts(ros: [Sequence[StepSequence], None],
                           concat_ros: [StepSequence, None],
                           batch_size: int) -> int:
    """
    Get the number of iterations from the given rollout data.

    :param ros: multiple rollouts
    :param concat_ros: concatenated rollouts
    :param batch_size: number of samples per batch
    :return: number of iterations (e.g. used for the progress bar)
    """
    if ros is None:
        assert concat_ros is not None
        return (concat_ros.length + batch_size - 1)//batch_size
    else:
        return (sum(ro.length for ro in ros) + batch_size - 1)//batch_size


def get_grad_via_torch(x_np: np.ndarray, fcn_to: Callable, *args_to, **kwargs_to) -> np.ndarray:
    r"""
    Get the gradient of a function operating on PyTorch tensors, by casting the input `x_np` as well as the
    resulting gradient to PyTorch.

    :param x_np: input vector $x$
    :param fcn_to: function $f(x, \cdot)$
    :param args_to: other arguments to the function
    :param kwargs_to: other keyword arguments to the function
    :return: $\nabla_x f(x, \cdot)$
    """
    if not isinstance(x_np, np.ndarray):
        raise pyrado.TypeErr(given=x_np, expected_type=np.ndarray)

    x_to = to.from_numpy(x_np)
    x_to.requires_grad = True
    out_to = fcn_to(x_to, *args_to, **kwargs_to)
    grad_to = to.autograd.grad(outputs=out_to, inputs=x_to, grad_outputs=to.ones_like(out_to))
    grad_to = grad_to[0]  # computes and returns the sum of gradients of outputs w.r.t. the inputs; we only have one
    return grad_to.numpy()
