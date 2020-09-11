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
import joblib
import os.path as osp
import torch as to
from copy import deepcopy
from torch.distributions import Distribution
from typing import NamedTuple, Optional, Union

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


def save_prefix_suffix(obj, name: str, file_ext: str, save_dir: str, meta_info: Optional[dict]):
    """
    Save an arbitrary object object using a prefix or suffix, depending on the meta information.

    :param obj: object to save
    :param name: name of the object for saving
    :param file_ext: file extension, e.g. 'pt' for policies
    :param save_dir: directory to save in
    :param meta_info: meta information that can contain a pre- and/or suffix for altering the name
    """
    if not isinstance(name, str):
        raise pyrado.TypeErr(given=name, expected_type=str)
    if not (file_ext == 'pt' or file_ext == 'pkl'):
        raise pyrado.ValueErr(given=file_ext, eq_constraint='pt or pkl')
    if not osp.isdir(save_dir):
        raise pyrado.PathErr(given=save_dir)

    if meta_info is None:
        if file_ext == 'pt':
            to.save(obj, osp.join(save_dir, f"{name}.{file_ext}"))

        elif file_ext == 'pkl':
            joblib.dump(obj, osp.join(save_dir, f"{name}.{file_ext}"))

    else:
        if not isinstance(meta_info, dict):
            raise pyrado.TypeErr(given=meta_info, expected_type=dict)

        if file_ext == 'pt':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(obj, osp.join(save_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(obj, osp.join(save_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(obj, osp.join(save_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                to.save(obj, osp.join(save_dir, f"{name}.{file_ext}"))

        elif file_ext == 'pkl':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                joblib.dump(obj, osp.join(save_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                joblib.dump(obj, osp.join(save_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                joblib.dump(obj, osp.join(save_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                joblib.dump(obj, osp.join(save_dir, f"{name}.{file_ext}"))


def load_prefix_suffix(obj, name: str, file_ext: str, load_dir: str, meta_info: Optional[dict]):
    """
    Load an arbitrary object object using a prefix or suffix, depending on the meta information.

    :param obj: object to load into
    :param name: name of the object for loading
    :param file_ext: file extension, e.g. 'pt' for policies
    :param load_dir: directory to load from
    :param meta_info: meta information that can contain a pre- and/or suffix for altering the name
    """
    if not isinstance(name, str):
        raise pyrado.TypeErr(given=name, expected_type=str)
    if not (file_ext == 'pt' or file_ext == 'pkl'):
        raise pyrado.ValueErr(given=file_ext, eq_constraint='pt or pkl')
    if not osp.isdir(load_dir):
        raise pyrado.PathErr(given=load_dir)

    if meta_info is None:
        if file_ext == 'pt':
            obj.load_state_dict(to.load(osp.join(load_dir, f"{name}.{file_ext}")).state_dict())

        elif file_ext == 'pkl':
            obj = joblib.load(osp.join(load_dir, f"{name}.{file_ext}"))

    else:
        if not isinstance(meta_info, dict):
            raise pyrado.TypeErr(given=meta_info, expected_type=dict)

        if file_ext == 'pt':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                obj.load_state_dict(to.load(osp.join(
                    load_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}")
                ).state_dict())
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                obj.load_state_dict(to.load(osp.join(
                    load_dir, f"{meta_info['prefix']}_{name}.{file_ext}")
                ).state_dict())
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                obj.load_state_dict(to.load(osp.join(
                    load_dir, f"{name}_{meta_info['suffix']}.{file_ext}")
                ).state_dict())
            else:
                obj.load_state_dict(to.load(osp.join(load_dir, f"{name}.{file_ext}")).state_dict())

        if file_ext == 'pkl':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                obj = joblib.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                obj = joblib.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                obj = joblib.load(osp.join(load_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                obj = joblib.load(osp.join(load_dir, f"{name}.{file_ext}"))

    return obj


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
