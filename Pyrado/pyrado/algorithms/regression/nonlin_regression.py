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

from typing import Optional

import numpy as np
import torch as to
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.utils.data_types import merge_dicts


class NonlinRegression(Algorithm):
    """ Train a policy using stochastic gradient descent to approximate the given data. """

    name: str = "regr"  # unique identifier

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        inputs: to.Tensor,
        targets: to.Tensor,
        policy: Policy,
        max_iter: int,
        max_iter_no_improvement: int = 30,
        optim_class=optim.Adam,
        optim_hparam: dict = None,
        loss_fcn=nn.MSELoss(),
        batch_size: int = 256,
        ratio_train: float = 0.8,
        max_grad_norm: Optional[float] = None,
        lr_scheduler=None,
        lr_scheduler_hparam: Optional[dict] = None,
        logger: StepLogger = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param inputs: input data set, where the samples are along the first dimension
        :param targets: target data set, where the samples are along the first dimension
        :param policy: Pyrado policy (subclass of PyTorch's Module) to train
        :param max_iter: maximum number of iterations
        :param max_iter_no_improvement: if the performance on the validation set did not improve for this many
                                        iterations, the policy is considered to have converged, i.e. training stops
        :param optim_class: PyTorch optimizer class
        :param optim_hparam: hyper-parameters for the PyTorch optimizer
        :param loss_fcn: loss function for training, by default `torch.nn.MSELoss()`
        :param batch_size: number of samples per policy update batch
        :param ratio_train: ratio of the training samples w.r.t. the total sample count
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(inputs, to.Tensor):
            raise pyrado.TypeErr(given=inputs, expected_type=to.Tensor)
        if not isinstance(targets, to.Tensor):
            raise pyrado.TypeErr(given=targets, expected_type=to.Tensor)
        if not isinstance(ratio_train, float):
            raise pyrado.TypeErr(given=ratio_train, expected_type=float)
        if not (0 < ratio_train < 1):
            raise pyrado.ValueErr(given=ratio_train, g_constraint="0", l_constraint="1")

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        # Construct the dataset (samples along rows)
        inputs = to.atleast_2d(inputs).T if inputs.ndimension() == 1 else inputs
        targets = to.atleast_2d(targets).T if targets.ndimension() == 1 else targets
        if inputs.shape[0] != targets.shape[0]:
            raise pyrado.ShapeErr(given=targets, expected_match=inputs)
        num_samples_all = inputs.shape[0]
        dataset = TensorDataset(inputs, targets)  # shared for training and validation loaders

        # Create training and validation loader
        idcs_all = to.randperm(num_samples_all)
        num_samples_trn = int(ratio_train * num_samples_all)
        num_samples_val = num_samples_all - num_samples_trn
        idcs_trn, idcs_val = idcs_all[:num_samples_trn], idcs_all[num_samples_trn:]
        self.loader_trn = DataLoader(
            dataset,
            batch_size=min(batch_size, num_samples_trn),
            drop_last=True,
            sampler=SubsetRandomSampler(idcs_trn),
        )
        self.loader_val = DataLoader(
            dataset,
            batch_size=min(batch_size, num_samples_val),
            drop_last=True,
            sampler=SubsetRandomSampler(idcs_val),
        )

        # Set defaults which can be overwritten by passing optim_hparam, and create the optimizer
        optim_hparam = merge_dicts([dict(lr=5e-3, eps=1e-8, weight_decay=1e-4), optim_hparam])
        self.optim = optim_class([{"params": self._policy.parameters()}], **optim_hparam)

        self.batch_size = batch_size
        self.ratio_train = ratio_train
        self.loss_fcn = loss_fcn
        self.max_grad_norm = max_grad_norm
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None and lr_scheduler_hparam is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

        # Stopping criterion
        self._curr_loss_val = pyrado.inf
        self._best_loss_val = pyrado.inf
        self._cnt_iter_no_improvement = 0
        self._max_iter_no_improvement = max_iter_no_improvement

    def stopping_criterion_met(self) -> bool:
        """
        Keep track of the best validation performance and check if it does not improve for a given number of iterations.

        :return: `True` if the performance on the validation set did not improve for, i.e. network has converged
        """
        if self._cnt_iter_no_improvement >= self._max_iter_no_improvement:
            # No improvement over on the validation set for self._max_iter_no_improvement iterations
            return True

        else:
            if self.curr_iter == 0 or self._curr_loss_val < self._best_loss_val:
                # Reset the counter if first epoch or any improvement
                self._best_loss_val = self._curr_loss_val
                self._cnt_iter_no_improvement = 0
            else:
                self._cnt_iter_no_improvement += 1

            # Continue training
            return False

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize the the stopping criterion's temporary variables
        self._curr_loss_val = -pyrado.inf
        self._best_loss_val = -pyrado.inf

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Update on the training set
        self._policy.train()
        losses_trn = []
        for batch in self.loader_trn:
            # Get the batch's data and predict
            inps, targs = batch[0], batch[1]
            preds = self._policy(inps)

            # Reset the gradients, compute and store the loss
            self.optim.zero_grad()
            loss_trn = self.loss_fcn(preds, targs)
            losses_trn.append(loss_trn)
            loss_trn.backward()

            # Clip the gradients if desired
            self.clip_grad(self._policy, self.max_grad_norm)

            # call the optimizer
            self.optim.step()

        # Calculate performance on the validation set
        self._policy.eval()
        losses_val = []
        with to.no_grad():
            for batch in self.loader_val:
                # Get the batch's data and predict
                inps, targs = batch[0], batch[1]
                preds = self._policy(inps)

                # Compute and store the loss
                losses_val.append(self.loss_fcn(preds, targs))

            # Log metrics computed from the old policy (loss value from update on this training sample)
            self._curr_loss_val = to.mean(to.as_tensor(losses_val))
            self.logger.add_value("avg trn loss", to.mean(to.as_tensor(losses_trn)), 6)
            self.logger.add_value("avg val loss", self._curr_loss_val, 6)
            self.logger.add_value(
                "min mag policy param", self._policy.param_values[to.argmin(abs(self._policy.param_values))]
            )
            self.logger.add_value(
                "max mag policy param", self._policy.param_values[to.argmax(abs(self._policy.param_values))]
            )
            if self._lr_scheduler is not None:
                self.logger.add_value("avg lr", np.mean(self._lr_scheduler.get_last_lr()), 6)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, -self._curr_loss_val.item(), meta_info)

    def save_snapshot(self, meta_info: Optional[dict] = None):
        super().save_snapshot()

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._policy, "policy.pt", self.save_dir, use_state_dict=True)
            if self.curr_iter == 0:
                pyrado.save(self.loader_trn, "loader_trn.pt", self.save_dir)
                pyrado.save(self.loader_val, "loader_val.pt", self.save_dir)
        else:
            # This algorithm instance is a subroutine of another algorithm
            pyrado.save(
                self._policy,
                "policy.pt",
                self.save_dir,
                prefix=meta_info.get("prefix", ""),
                suffix=meta_info.get("suffix", ""),
                use_state_dict=True,
            )
            if self.curr_iter == 0:
                pyrado.save(
                    self.loader_trn,
                    "loader_trn.pt",
                    self.save_dir,
                    prefix=meta_info.get("prefix", ""),
                    suffix=meta_info.get("suffix", ""),
                )
                pyrado.save(
                    self.loader_trn,
                    "loader_val.pt",
                    self.save_dir,
                    prefix=meta_info.get("prefix", ""),
                    suffix=meta_info.get("suffix", ""),
                )
