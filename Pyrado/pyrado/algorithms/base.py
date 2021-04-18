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

import joblib
import os.path as osp
from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.step_sequence import StepSequence
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pyrado
from pyrado.logger.experiment import split_path_custom_common
from pyrado.exploration.stochastic_action import StochasticActionExplStrat
from pyrado.exploration.stochastic_params import StochasticParamExplStrat
from pyrado.logger.step import StepLogger, LoggerAware
from pyrado.policies.base import Policy
from pyrado import set_seed
from pyrado.utils import get_class_name
from pyrado.utils.input_output import print_cbt


class Algorithm(ABC, LoggerAware):
    """
    Base class of all algorithms in Pyrado
    Algorithms specify the way how the policy is updated as well as the exploration strategy used to acquire samples.
    """

    name: str = None  # unique identifier
    iteration_key: str = "iteration"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        max_iter: int,
        policy: Optional[Policy],
        logger: Optional[StepLogger] = None,
        save_name: str = "algo",
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param policy: Pyrado policy (subclass of PyTorch's Module) to train
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param save_name: name of the algorithm's pickle file without the ending, this becomes important if the
                          algorithm is run as a subroutine
        """
        if not isinstance(max_iter, int) and max_iter > 0:
            raise pyrado.ValueErr(given=max_iter, g_constraint="0")
        if not isinstance(policy, Policy) and policy is not None:
            raise pyrado.TypeErr(msg="If a policy is given, it needs to be of type Policy!")
        if not isinstance(logger, StepLogger) and logger is not None:
            raise pyrado.TypeErr(msg="If a logger is given, it needs to be of type StepLogger!")
        if not isinstance(save_name, str):
            raise pyrado.TypeErr(given=save_name, expected_type=str)

        self._save_dir = save_dir
        self._save_name = save_name
        self._max_iter = max_iter
        self._curr_iter = 0
        self._policy = policy
        self._logger = logger
        self._cnt_samples = 0
        self._highest_avg_ret = -pyrado.inf  # for snapshot_mode = 'best'

    @property
    def save_dir(self) -> str:
        """ Get the directory where the data is saved to. """
        return self._save_dir

    @save_dir.setter
    def save_dir(self, save_dir: pyrado.PathLike):
        """ Set the directory where the data is saved to. """
        if not osp.isdir(save_dir):
            raise pyrado.PathErr(given=save_dir)
        self._save_dir = save_dir

    @property
    def save_name(self) -> str:
        """ Get the name for saving this algorithm instance, e.g. 'algo' if saved to 'algo.pkl'. """
        return self._save_name

    @save_name.setter
    def save_name(self, name: str):
        """ Set the name for saving this algorithm instance, e.g. 'subrtn' if saved to 'subrtn.pkl'. """
        if not isinstance(name, str):
            raise pyrado.TypeErr(given=name, expected_type=str)
        self._save_name = name

    @property
    def max_iter(self) -> int:
        """ Get the maximum number of iterations. """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        """ Set the maximum number of iterations. """
        assert max_iter > 0
        self._max_iter = max_iter

    @property
    def curr_iter(self) -> int:
        """ Get the current iteration counter. """
        return self._curr_iter

    @property
    def sample_count(self) -> int:
        """ Get the total number of samples, i.e. steps of a rollout, used for training so far. """
        return self._cnt_samples

    @property
    def policy(self) -> Policy:
        """ Get the algorithm's policy. """
        return self._policy

    @property
    def expl_strat(self) -> Union[StochasticActionExplStrat, StochasticParamExplStrat, None]:
        """ Get the algorithm's exploration strategy. """
        return None

    def stopping_criterion_met(self) -> bool:
        """
        Checks if one of the algorithms (characteristic) stopping criteria is met.

        .. note::
            This function can be overwritten by the subclasses to implement custom stopping behavior.

        :return: flag if one of the stopping criterion(s) is met
        """
        return False

    def reset(self, seed: int = None):
        """
        Reset the algorithm to it's initial state. This should NOT reset learned policy parameters.
        By default, this resets the iteration count and the exploration strategy.
        Be sure to call this function if you override it.

        :param seed: seed value for the random number generators, pass `None` for no seeding
        """
        # Reset the exploration strategy if any
        if self.expl_strat is not None:
            self.expl_strat.reset_expl_params()

        # Reset internal variables
        self._curr_iter = 0
        self._cnt_samples = 0
        self._highest_avg_ret = -pyrado.inf

        # Set all rngs' seeds
        if seed is not None:
            set_seed(seed, verbose=True)

    def init_modules(self, warmstart: bool, suffix: str = "", prefix: str = None, **kwargs):
        """
        Initialize the algorithm's learnable modules, e.g. a policy or value function.
        Overwrite this method if the algorithm uses a learnable module aside the policy, e.g. a value function.

        :param warmstart: if `True`, the algorithm starts learning with an initialization. This can either be the a
                          fixed parameter vector, or the results of the previous iteration
        :param suffix: keyword for `meta_info` when loading from previous iteration
        :param prefix: keyword for `meta_info` when loading from previous iteration
        :param kwargs: keyword arguments for initialization, e.g. `policy_param_init` or `valuefcn_param_init`
        """
        if prefix is None:
            prefix = f"iter_{self._curr_iter - 1}"

        ppi = kwargs.get("policy_param_init", None)

        if warmstart and ppi is not None:
            self._policy.init_param(ppi)
            print_cbt("Learning given an fixed parameter initialization.", "w")

        elif warmstart and ppi is None and self._curr_iter > 0:
            self._policy = pyrado.load("policy.pt", self.save_dir, prefix=prefix, suffix=suffix, obj=self._policy)
            print_cbt(f"Learning given the results from iteration {self._curr_iter - 1}", "w")

        else:
            # Reset the policy
            self._policy.init_param()
            print_cbt("Learning from scratch.", "w")

    def train(self, snapshot_mode: str = "latest", seed: int = None, meta_info: dict = None):
        """
        Train one/multiple policy/policies in a given environment.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new high-score)
        :param seed: seed value for the random number generators, pass `None` for no seeding
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        if self._policy is not None:
            print_cbt(
                f"{get_class_name(self)} started training a {get_class_name(self._policy)} "
                f"with {self._policy.num_param} parameters using the snapshot mode {snapshot_mode}.",
                "g",
            )
            # Set dropout and batch normalization layers to training mode
            self._policy.train()
        else:
            print_cbt(f"{get_class_name(self)} started training using the snapshot mode {snapshot_mode}.", "g")

        # Set all rngs' seeds
        if seed is not None:
            set_seed(seed, verbose=True)

        while self._curr_iter < self.max_iter and not self.stopping_criterion_met():
            # Record current iteration to logger
            self.logger.add_value(self.iteration_key, self._curr_iter)

            # Acquire data, save the training progress, and update the parameters
            self.step(snapshot_mode, meta_info)

            # Update logger and print
            self.logger.record_step()

            # Increase the iteration counter
            self._curr_iter += 1

        if self.stopping_criterion_met():
            stopping_reason = "Stopping criterion met!"
        else:
            stopping_reason = "Maximum number of iterations reached!"

        if self._policy is not None:
            print_cbt(
                f"{get_class_name(self)} finished training a {get_class_name(self._policy)} "
                f"with {self._policy.num_param} parameters. {stopping_reason}",
                "g",
            )
            # Set dropout and batch normalization layers to evaluation mode
            self._policy.eval()
        else:
            print_cbt(f"{get_class_name(self)} finished training. {stopping_reason}", "g")

    @abstractmethod
    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Perform a single iteration of the algorithm. This includes collecting the data, updating the parameters, and
        adding the metrics of interest to the logger. Does not update the `curr_iter` attribute.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any):
        """ Update the policy's (and value functions') parameters based on the collected rollout data. """
        pass

    def make_snapshot(self, snapshot_mode: str, curr_avg_ret: float = None, meta_info: dict = None):
        """
        Make a snapshot of the training progress.
        This method is called from the subclasses and delegates to the custom method `save_snapshot()`.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param curr_avg_ret: current average return used for the snapshot_mode 'best' to trigger `save_snapshot()`
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        if snapshot_mode == "latest":
            self.save_snapshot(meta_info)
        elif snapshot_mode == "best":
            if curr_avg_ret is None:
                raise pyrado.ValueErr(msg="curr_avg_ret must not be None when snapshot_mode = 'best'!")
            if curr_avg_ret > self._highest_avg_ret:
                self._highest_avg_ret = curr_avg_ret
                self.save_snapshot(meta_info)
        elif snapshot_mode in {"no", "None"}:
            pass  # don't save anything
        else:
            raise pyrado.ValueErr(given=snapshot_mode, eq_constraint="'latest', 'best', or 'no'")

    def save_snapshot(self, meta_info: dict = None):
        """
        Save the algorithm information (e.g., environment, policy, ect.).
        Subclasses should call the base method to save the policy.

        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a `dict` of information about the current iteration of the meta-algorithm
        """
        joblib.dump(self, osp.join(self.save_dir, f"{self._save_name}.pkl"))

    @staticmethod
    def load_snapshot(load_dir: pyrado.PathLike, load_name: str = "algo"):
        """
        Load an algorithm from file, i.e. unpickle it.

        :param load_dir: experiment directory to load from
        :param load_name: name of the algorithm's pickle file without the ending
        """
        if not osp.isdir(load_dir):
            raise pyrado.PathErr(given=load_dir)

        file = osp.join(load_dir, f"{load_name}.pkl")
        if not osp.isfile(file):
            raise pyrado.PathErr(given=file)

        algo = joblib.load(file)

        if not isinstance(algo, Algorithm):
            raise pyrado.TypeErr(given=algo, expected_type=Algorithm)

        return algo

    @staticmethod
    def clip_grad(module: nn.Module, max_grad_norm: Optional[float]) -> float:
        """
        Clip all gradients of the provided Module (e.g., a policy or an advantage estimator) by their L2 norm value.

        .. note::
            The gradient clipping has to be applied between loss.backward() and optimizer.step()

        :param module: Module containing parameters
        :param max_grad_norm: maximum L2 norm for the gradient
        :return: total norm of the parameters (viewed as a single vector)
        """
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(module.parameters(), max_grad_norm, norm_type=2)  # returns unclipped norm

        # Calculate the clipped gradient's L2 norm (for logging)
        total_norm = 0.0
        for p in list(filter(lambda p: p.grad is not None, module.parameters())):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def __getstate__(self):
        # Disassemble the directory on pickling
        _, common_part = split_path_custom_common(self._save_dir)
        self.__dict__["_save_dir_common"] = common_part
        return self.__dict__

    def __setstate__(self, state):
        # Assemble the directory on unpickling
        self.__dict__ = state
        common_part = state["_save_dir_common"]

        # First, try if it has been split at pyrado.EXP_DIR
        self._save_dir = osp.join(pyrado.EXP_DIR, common_part)
        if not osp.isdir(self._save_dir):
            # If that did not work, try if it has been split at pyrado.TEMP_DIR
            self._save_dir = osp.join(pyrado.TEMP_DIR, common_part)
            if not osp.isdir(self._save_dir):
                # If that did not work, try if it has been split at the pytest's temporary path
                self._save_dir = osp.join("/tmp", common_part)
                if not osp.isdir(self._save_dir):
                    raise pyrado.PathErr(given=self._save_dir)


class InterruptableAlgorithm(Algorithm, ABC):
    """
    A simple checkpoint system too keep track of the algorithms progress. The cyclic counter starts at `init_checkpoint`
    and counts until (including) `num_checkpoints`, and is then reset to zero.
    """

    def __init__(self, num_checkpoints: int, init_checkpoint: int = 0, *args, **kwargs):
        """
        Constructor

        :param num_checkpoints: total number of checkpoints
        :param init_checkpoint: initial value of the cyclic counter, defaults to 0, use negative values can to mark
                                sections that should only be executed once
        :param args: positional arguments forwarded to Algorithm's constructor
        :param kwargs: keyword arguments forwarded to Algorithm's constructor
        """
        if not isinstance(num_checkpoints, int):
            raise pyrado.TypeErr(given=num_checkpoints, expected_type=int)
        if num_checkpoints < 1:
            raise pyrado.ValueErr(given=num_checkpoints, ge_constraint="1")
        if not isinstance(init_checkpoint, int):
            raise pyrado.TypeErr(given=init_checkpoint, expected_type=int)

        self._num_checkpoints = num_checkpoints
        self._curr_checkpoint = init_checkpoint

        # Call Algorithm's constructor
        super().__init__(*args, **kwargs)

    @property
    def curr_checkpoint(self) -> int:
        """ Get the current checkpoint counter. """
        return self._curr_checkpoint

    def reset_checkpoint(self, curr: int = 0):
        """
        Explicitly reset the cyclic counter.

        :param curr: value to set the counter to, defaults to 0
        """
        if not isinstance(curr, int):
            raise pyrado.TypeErr(given=curr, expected_type=int)
        self._curr_checkpoint = curr

    def reset(self, seed: int = None):
        super().reset(seed)
        self.reset_checkpoint()

    def reached_checkpoint(self, meta_info: dict = None):
        """
        Increase the cyclic counter by 1. When the counter reached the maximum number of checkpoints, defined in the
        constructor, it is automatically reset to zero.
        This method also saves the algorithm instance using `save_snapshot()`, otherwise increasing the checkpoint
        counter can have no effect.

        :param meta_info: information forwarded to `save_snapshot()`
        """
        next = self._curr_checkpoint + 1
        self._curr_checkpoint = next % (self._num_checkpoints + 1) if next > 0 else next  # no modulo for negative count

        self.save_snapshot(meta_info)


class ExposedSampler:
    """A mixin class indicating that this algorithm exposes its sampler.

    Implementors: Save the used sampler in the `self.sampler` property.
    """

    @property
    def sampler(self) -> SamplerBase:
        """Returns the sampler of the algorithm

        :return: The (initialized) sampler used by the algorithm
        :rtype: SamplerBase
        """
        return self.sampler

    def sample(self, *args, **kwargs) -> List[StepSequence]:
        """Calls the sample method of the algorithm's sampler.

        :param *args: Arguments to be forwarded to the sample method
        :param **kwargs: Keyword-Arguments to be forwarded to the sample method
        :return: A list of `StepSequence`s, which are generated according to the algorithms parameters (e.g. number of workers, rollout length, ...)
        :rtype: List[StepSequence]
        """
        return self.sampler.sample(*args, **kwargs)
