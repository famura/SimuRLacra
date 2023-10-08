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

import sys
from contextlib import ExitStack
from typing import Optional, Sequence, Union

import numpy as np
import torch as to
import torch.nn as nn
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import num_iter_from_rollouts
from pyrado.logger.step import LoggerAware
from pyrado.policies.base import Policy
from pyrado.policies.recurrent.base import RecurrentPolicy
from pyrado.sampling.step_sequence import StepSequence, discounted_values
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_processing import RunningStandardizer, standardize
from pyrado.utils.math import explained_var


class GAE(LoggerAware, nn.Module):
    """
    General Advantage Estimation (GAE)

    .. seealso::
        [1] J. Schulmann, P. Moritz, S. Levine, M. Jordan, P. Abbeel, 'High-Dimensional Continuous Control Using
        Generalized Advantage Estimation', ICLR 2016
    """

    def __init__(
        self,
        vfcn: Union[nn.Module, Policy],
        gamma: float = 0.99,
        lamda: float = 0.95,
        num_epoch: int = 10,
        batch_size: int = 64,
        standardize_adv: bool = True,
        standardizer: Optional[RunningStandardizer] = None,
        max_grad_norm: Optional[float] = None,
        lr: float = 5e-4,
        lr_scheduler=None,
        lr_scheduler_hparam: Optional[dict] = None,
    ):
        r"""
        Constructor

        :param vfcn: value function, which can be a `FNN` or a `Policy`
        :param gamma: temporal discount factor
        :param lamda: regulates the trade-off between bias (max for 0) and variance (max for 1), see [1]
        :param num_epoch: number of iterations over all gathered samples during one estimator update
        :param batch_size: number of samples per estimator update batch
        :param standardize_adv: if `True`, the advantages are standardized to be $~ N(0,1)$
        :param standardizer: pass `None` to use stateless standardisation, alternatively pass `RunningStandardizer()`
                             to use a standardizer wich keeps track of past values
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        """
        if not isinstance(vfcn, (nn.Module, Policy)):
            raise pyrado.TypeErr(given=vfcn, expected_type=[nn.Module, Policy])
        if isinstance(vfcn, Policy):
            if not vfcn.env_spec.act_space == ValueFunctionSpace:
                raise pyrado.ShapeErr(msg="The given act_space held by the vfcn should be a ValueFunctionSpace.")
        if not 0 <= gamma <= 1:
            raise pyrado.ValueErr(given=gamma, ge_constraint="0", le_constraint="1")
        if not 0 <= lamda <= 1:
            raise pyrado.ValueErr(given=lamda, ge_constraint="0", le_constraint="1")

        # Call Module's constructor
        super().__init__()

        # Store the inputs
        self._vfcn = vfcn
        self.gamma = gamma
        self.lamda = lamda
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.standardize_adv = standardize_adv
        self.standardizer = standardizer

        # Initialize
        self.loss_fcn = nn.MSELoss()
        self.optim = to.optim.Adam(self._vfcn.parameters(), lr=lr, eps=1e-5)
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    @property
    def vfcn(self) -> Union[nn.Module, Policy]:
        """Get the value function approximator."""
        return self._vfcn

    @vfcn.setter
    def vfcn(self, vfcn: Union[nn.Module, Policy]):
        """Set the value function approximator."""
        if not isinstance(vfcn, (nn.Module, Policy)):
            raise pyrado.TypeErr(given=vfcn, expected_type=[nn.Module, Policy])
        self._vfcn = vfcn

        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1

    def gae(
        self, concat_ros: StepSequence, v_pred: Optional[to.Tensor] = None, requires_grad: bool = False
    ) -> to.Tensor:
        """
        Compute the generalized advantage estimation as described in [1].

        :param concat_ros: concatenated rollouts (sequence of steps from potentially different rollouts)
        :param v_pred: state-value predictions if already computed, else pass None
        :param requires_grad: is the gradient required
        :return adv: tensor of advantages
        """
        with ExitStack() as stack:
            if not requires_grad:
                stack.enter_context(to.no_grad())
            if v_pred is None:
                # Get the predictions from the value function
                v_pred = self.values(concat_ros)

            # Compute the advantages
            adv = to.empty_like(v_pred)
            for k in reversed(range(concat_ros.length)):
                if concat_ros[k].done:
                    adv[k] = concat_ros[k].reward - v_pred[k]
                else:
                    adv[k] = (
                        concat_ros[k].reward
                        + self.gamma * v_pred[k + 1]
                        - v_pred[k]
                        + self.gamma * self.lamda * adv[k + 1]
                    )

            if self.standardize_adv:
                if isinstance(self.standardizer, RunningStandardizer):
                    adv = self.standardizer(adv, axis=0)
                else:
                    adv = standardize(adv)

            return adv

    def tdlamda_returns(
        self, v_pred: to.Tensor = None, adv: to.Tensor = None, concat_ros: StepSequence = None
    ) -> to.Tensor:
        r"""
        Compute the TD($\lambda$) returns based on the predictions of the network (introduces a bias).

        :param v_pred: state-value predictions if already computed, pass `None` to compute form given rollouts
        :param adv: advantages if already computed, pass `None` to compute form given rollouts
        :param concat_ros: rollouts to compute predicted values and advantages from if they are not provided
        :return: exponentially weighted returns based on the value function estimator
        """
        with to.no_grad():
            if v_pred is None:
                if concat_ros is None:
                    raise pyrado.TypeErr(given=concat_ros, expected_type=StepSequence)
                v_pred = self.values(concat_ros)
            if adv is None:
                if concat_ros is None:
                    raise pyrado.TypeErr(given=concat_ros, expected_type=StepSequence)
                adv = self.gae(concat_ros, v_pred)

            # Return the (bootstrapped) target for the value function prediction
            return v_pred + adv  # Q = V + A

    def values(self, concat_ros: StepSequence) -> to.Tensor:
        """
        Compute the states' values for all observations.

        :param concat_ros: concatenated rollouts
        :return: states' values
        """
        if isinstance(self._vfcn, Policy):
            # Use the Policy's forward method and the hidden states if they have been saved during the rollout
            v_pred = self._vfcn.evaluate(concat_ros, hidden_states_name="vf_hidden_states")
        else:
            v_pred = self._vfcn(concat_ros.observations)  # not a recurrent network
        return v_pred

    def update(self, rollouts: Sequence[StepSequence], use_empirical_returns: bool = False):
        """
        Adapt the parameters of the advantage function estimator, minimizing the MSE loss for the given samples.

        :param rollouts: batch of rollouts
        :param use_empirical_returns: use the return from the rollout (True) or the ones from the V-fcn (False)
        :return adv: tensor of advantages after V-function updates
        """
        # Turn the batch of rollouts into a list of steps
        concat_ros = StepSequence.concat(rollouts)
        concat_ros.torch(data_type=to.get_default_dtype())

        if use_empirical_returns:
            # Compute the value targets (empirical discounted returns) for all samples
            v_targ = discounted_values(rollouts, self.gamma).view(-1, 1)
        else:
            # Use the value function to compute the value targets (also called bootstrapping)
            v_targ = self.tdlamda_returns(concat_ros=concat_ros)
        concat_ros.add_data("v_targ", v_targ)

        # Logging
        with to.no_grad():
            v_pred_old = self.values(concat_ros)
            loss_old = self.loss_fcn(v_pred_old, v_targ)
        vfcn_grad_norm = []

        # Iterate over all gathered samples num_epoch times
        for e in range(self.num_epoch):
            for batch in tqdm(
                concat_ros.split_shuffled_batches(
                    self.batch_size, complete_rollouts=isinstance(self.vfcn, RecurrentPolicy)
                ),
                total=num_iter_from_rollouts(None, concat_ros, self.batch_size),
                desc=f"Epoch {e}",
                unit="batches",
                file=sys.stdout,
                leave=False,
            ):
                # Reset the gradients
                self.optim.zero_grad()

                # Make predictions for this mini-batch using values function
                v_pred = self.values(batch)

                # Compute estimator loss for this mini-batch and backpropagate
                vfcn_loss = self.loss_fcn(v_pred, batch.v_targ)
                vfcn_loss.backward()

                # Clip the gradients if desired
                vfcn_grad_norm.append(Algorithm.clip_grad(self.vfcn, self.max_grad_norm))

                # Call optimizer
                self.optim.step()

            # Update the learning rate if a scheduler has been specified
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

        # Estimate the advantage after fitting the parameters of the V-fcn
        adv = self.gae(concat_ros)  # is done with to.no_grad()

        with to.no_grad():
            v_pred_new = self.values(concat_ros)
            loss_new = self.loss_fcn(v_pred_new, v_targ)
            vfcn_loss_impr = loss_old - loss_new  # positive values are desired
            explvar = explained_var(v_pred_new, v_targ)  # values close to 1 are desired

        # Log metrics computed from the old value function (before the update)
        self.logger.add_value("explained var critic", explvar, 4)
        self.logger.add_value("loss improv critic", vfcn_loss_impr, 4)
        self.logger.add_value("avg grad norm critic", np.mean(vfcn_grad_norm), 4)
        if self._lr_scheduler is not None:
            self.logger.add_value("lr critic", np.mean(self._lr_scheduler.get_last_lr()), 6)

        return adv

    def reset(self):
        """
        Reset the advantage estimator to it's initial state.
        The default implementation resets the learning rate scheduler if there is one.
        """
        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1
