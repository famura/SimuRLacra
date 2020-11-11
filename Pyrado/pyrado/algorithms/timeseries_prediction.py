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

import numpy as np
import torch as to
import torch.nn as nn
from torch import optim
from typing import Optional

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.utils.saving_loading import save_prefix_suffix
from pyrado.utils.data_sets import TimeSeriesDataSet
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.recurrent.rnn import RNNPolicyBase
from pyrado.utils.input_output import print_cbt


class TSPred(Algorithm):
    """ Train a policy to predict a time series of data. """

    name: str = 'tspred'  # unique identifier

    def __init__(self,
                 save_dir: str,
                 dataset: TimeSeriesDataSet,
                 policy: Policy,
                 max_iter: int,
                 windowed: bool = False,
                 cascaded: bool = False,
                 optim_class=optim.Adam,
                 optim_hparam: dict = None,
                 loss_fcn=nn.MSELoss(),
                 lr_scheduler=None,
                 lr_scheduler_hparam: Optional[dict] = None,
                 num_workers: int = 1,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param dataset: complete data set, where the samples are along the first dimension
        :param policy: Pyrado policy (subclass of PyTorch's Module) to train
        :param max_iter: maximum number of iterations
        :param windowed: if `True`, one fixed-length (short) input sequence is provided to the policy which then
                         predicts one sample, else the complete (long) input sequence is fed to the policy which then
                         predicts an sequence of samples of equal length
        :param cascaded: it `True`, the predictions are made based on previous predictions instead of the current input
        :param optim_class: PyTorch optimizer class
        :param optim_hparam: hyper-parameters for the PyTorch optimizer
        :param loss_fcn: loss function for training, by default `torch.nn.MSELoss()`
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(dataset, TimeSeriesDataSet):
            raise pyrado.TypeErr(given=dataset, expected_type=TimeSeriesDataSet)
        if not policy.is_recurrent:
            raise pyrado.TypeErr(msg='TSPred algorithm only supports recurrent policies!')

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        # Store the inputs
        self.dataset = dataset
        self.cascaded = cascaded
        self.windowed = windowed
        self.loss_fcn = loss_fcn

        optim_hparam = dict(lr=1e-1, eps=1e-8, weight_decay=1e-4) if optim_hparam is None else optim_hparam
        self.optim = optim_class([{'params': self._policy.parameters()}], **optim_hparam)
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None and lr_scheduler_hparam is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    def step(self, snapshot_mode: str, meta_info: dict = None):

        # Feed one epoch of the training set to the policy
        if self.windowed:
            # Predict
            loss_trn = []
            for inp_seq, targ in self.dataset.data_trn_seqs:
                # Reset the gradients
                self.optim.zero_grad()

                # Predict
                pred_trn = TSPred.predict(self._policy, inp_seq, self.windowed, self.cascaded)[0]
                targ_trn = targ.unsqueeze(0)

                # Compute the loss, backpropagate, and call the optimizer
                loss_trn_single = self.loss_fcn(targ_trn, pred_trn)
                loss_trn_single.backward()
                self.optim.step()
                loss_trn.append(loss_trn_single.item())
            loss_trn = to.mean(to.tensor(loss_trn))

        elif not self.windowed:
            # Reset the gradients
            self.optim.zero_grad()

            # Predict
            preds_trn = TSPred.predict(self._policy, self.dataset.data_trn_inp, self.windowed, self.cascaded)[0]
            targs_trn = self.dataset.data_trn_targ

            # Compute the loss, backpropagate, and call the optimizer
            loss_trn = self.loss_fcn(targs_trn, preds_trn)
            loss_trn.backward()
            self.optim.step()

        # Update the learning rate if a scheduler has been specified
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        # Logging the loss on the test set
        with to.no_grad():
            # Feed one epoch of the testing set to the policy
            if self.windowed:
                preds_tst = to.stack([TSPred.predict(self._policy, inp_seq, self.windowed, self.cascaded)[0]
                                      for inp_seq, _ in self.dataset.data_tst_seqs])
                targs_tst = to.stack([targ.unsqueeze(0) for _, targ in self.dataset.data_tst_seqs])
            elif not self.windowed:
                preds_tst = TSPred.predict(self._policy, self.dataset.data_tst_inp, self.windowed, self.cascaded)[0]
                targs_tst = self.dataset.data_tst_targ
            loss_tst = self.loss_fcn(targs_tst, preds_tst)

        # Log metrics computed from the old/updated policy (loss value from update on this training sample)
        self.logger.add_value('trn loss', loss_trn, 6)
        self.logger.add_value('tst loss', loss_tst, 6)
        self.logger.add_value('min mag policy param',
                              self._policy.param_values[to.argmin(abs(self._policy.param_values))])
        self.logger.add_value('max mag policy param',
                              self._policy.param_values[to.argmax(abs(self._policy.param_values))])
        if self._lr_scheduler is not None:
            self.logger.add_value('avg learning rate', np.mean(self._lr_scheduler.get_lr()), 6)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, -loss_trn.item(), meta_info)

    @staticmethod
    def predict(policy: Policy,
                inp_seq: to.Tensor,
                windowed: bool,
                cascaded: bool,
                hidden: Optional[to.Tensor] = None) -> (to.Tensor, to.Tensor):
        """
        Reset the hidden states, predict one output given a arbitrary long sequence of inputs.

        :param policy: policy used to make the predictions
        :param inp_seq: input sequence
        :param hidden: initial hidden states, pass `None` to let the network pick its default hidden state
        :param windowed: if `True`, one fixed-length (short) input sequence is provided to the policy which then
                         predicts one sample, else the complete (long) input sequence is fed to the policy which then
                         predicts an sequence of samples of equal length
        :param cascaded: it `True`, the predictions are made based on previous predictions instead of the current input
        :return: predicted output and latest hidden state
        """
        # Custom RNNs
        if isinstance(policy, PotentialBasedPolicy):
            preds = []
            last_pred = None

            # Run steps consecutively reusing the hidden state
            for idx, inp in enumerate(inp_seq):
                if cascaded and idx > 0:
                    # Use the latest prediction instead of the current input
                    inp = last_pred
                last_pred, hidden = policy(inp, hidden)  # hidden is unpacked at the beginning and packed at the end
                preds.append(last_pred)

            if windowed:
                # Select the latest (hidden is already the latest)
                preds = preds[-1].view(1, -1)
            else:
                preds = to.stack(preds)

        # PyTorch RNNs
        elif isinstance(policy, RNNPolicyBase):
            if cascaded:
                preds = []
                last_pred = None

                # Run steps consecutively reusing the hidden state
                for idx, inp in enumerate(inp_seq):
                    if idx > 0:
                        # Use the latest prediction instead of the current input
                        inp = last_pred
                    last_pred, hidden = policy(inp, hidden)  # hidden is unpacked at the beginning and packed at the end
                    preds.append(last_pred)

                # Return the (latest if windowed) prediction and hidden state
                if windowed:
                    preds = preds[-1].view(1, -1)
                else:
                    preds = to.stack(preds)

            else:
                # Pass all inputs at once
                preds, hidden = policy(inp_seq, hidden)

                # Select the latest
                if windowed:
                    preds = preds[-1].view(1, -1)
                hidden = hidden[-1].view(1, -1)

        else:
            raise pyrado.TypeErr(given=policy, expected_type=[PotentialBasedPolicy, RNNPolicyBase])

        # Return the (latest if windowed) prediction and the latest hidden state
        return preds, hidden

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot()

        # Does not matter if this algorithm instance is a subroutine of another algorithm
        save_prefix_suffix(self._policy, 'policy', 'pt', self.save_dir, meta_info)
        save_prefix_suffix(self.dataset, 'dataset', 'pt', self.save_dir, meta_info)

    @staticmethod
    def evaluate(policy: Policy,
                 inps: to.Tensor,
                 targs: to.Tensor,
                 windowed: bool,
                 cascaded: bool,
                 num_init_samples: int,
                 hidden: Optional[to.Tensor] = None,
                 loss_fcn=nn.MSELoss(),
                 verbose: bool = True):
        if not inps.shape[0] == targs.shape[0]:
            raise pyrado.ShapeErr(given=inps, expected_match=targs)

        policy.eval()
        targs = targs[num_init_samples:, :] if num_init_samples > 0 else targs
        preds = to.empty_like(targs)

        # Pass the first samples through the network in order to initialize the hidden state
        inp = inps[:num_init_samples, :] if num_init_samples > 0 else inps[0].unsqueeze(0)  # running input
        pred, hidden = TSPred.predict(policy, inp, windowed, cascaded=False, hidden=hidden)

        # Run steps consecutively reusing the hidden state
        for idx in range(inps.shape[0] - num_init_samples):
            if not cascaded or idx == 0:
                # Forget the oldest input and append the latest input
                inp = inps[idx + num_init_samples, :].unsqueeze(0)
            else:
                # Forget the oldest input and append the latest prediction
                inp = pred

            pred, hidden = TSPred.predict(policy, inp, windowed, cascaded=False, hidden=hidden)
            preds[idx, :] = pred

        # Compute loss for the entire data set at once
        loss = loss_fcn(targs, preds)

        if verbose:
            print_cbt(
                f'The {policy.name} policy with {policy.num_param} parameters predicted {inps.shape[0]} data points '
                f'with a loss of {loss.item():.4e}.', 'g')

        return preds, loss
