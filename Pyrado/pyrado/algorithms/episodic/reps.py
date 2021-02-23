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
import sys
import torch as to
from functools import partial
from scipy import optimize
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from typing import Callable, Union, Optional

import pyrado
from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.algorithms.utils import get_grad_via_torch
from pyrado.logger.step import StepLogger
from pyrado.policies.special.domain_distribution import DomainDistrParamPolicy
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_params import NormalParamNoise, SymmParamExplStrat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.utils.input_output import print_cbt_once
from pyrado.utils.math import logmeanexp
from pyrado.policies.base import Policy
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult


class REPS(ParameterExploring):
    """
    Episodic variant of Relative Entropy Policy Search (REPS)

    .. note::
        REPS was designed for linear policies.

    .. seealso::
        [1] J. Peters, K. Mülling, Y. Altuen, "Relative Entropy Policy Search", AAAI, 2010
        [2] A. Abdolmaleki, J.T. Springenberg, J. Degrave, S. Bohez, Y. Tassa, D. Belov, N. Heess, M. Riedmiller,
            "Relative Entropy Regularized Policy Iteration", arXiv, 2018
        [3] This implementation is inspired by the work of H. Abdulsamad
            https://github.com/hanyas/rl/blob/master/rl/ereps/ereps.py
    """

    name: str = "reps"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        policy: Policy,
        max_iter: int,
        eps: float,
        num_init_states_per_domain: int,
        pop_size: Optional[int],
        expl_std_init: float,
        expl_std_min: float = 0.01,
        num_domains: Optional[int] = 1,
        symm_sampling: bool = False,
        num_epoch_dual: int = 1000,
        softmax_transform: bool = False,
        use_map: bool = True,
        optim_mode: str = "scipy",
        lr_dual: float = 5e-4,
        num_workers: int = 4,
        logger: Optional[StepLogger] = None,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param eps: bound on the KL divergence between policy updates, e.g. 0.1
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param pop_size: number of solutions in the population
        :param num_init_states_per_domain: number of rollouts to cover the variance over initial states
        :param num_domains: number of rollouts due to the variance over domain parameters
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param symm_sampling: use an exploration strategy which samples symmetric populations
        :param num_epoch_dual: number of epochs for the minimization of the dual function
        :param softmax_transform: pass `True` to use a softmax to transform the returns, else use a shifted exponential
        :param use_map: use maximum a-posteriori likelihood (`True`) or maximum likelihood (`False`) update rule
        :param optim_mode: choose the type of optimizer: 'torch' for a SGD-based optimizer or 'scipy' for optimizers
                           from scipy (here SLSQP)
        :param lr_dual: learning rate for the dual's optimizer, ignored for `optim_mode` `'scipy'`
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(policy, (LinearPolicy, DomainDistrParamPolicy)):
            print_cbt_once("REPS was designed for linear policies.", "y")

        # Call ParameterExploring's constructor
        super().__init__(
            save_dir=save_dir,
            env=env,
            policy=policy,
            max_iter=max_iter,
            num_init_states_per_domain=num_init_states_per_domain,
            num_domains=num_domains,
            pop_size=pop_size,
            num_workers=num_workers,
            logger=logger,
        )

        # Store the inputs
        self.eps = eps
        self.softmax_transform = softmax_transform
        self.use_map = use_map

        # Explore using normal noise
        self._expl_strat = NormalParamNoise(
            self._policy.num_param,
            full_cov=True,
            std_init=expl_std_init,
            std_min=expl_std_min,
            use_cuda=self._policy.device != "cpu",
        )
        if symm_sampling:
            # Exploration strategy based on symmetrical normally distributed noise
            if self.pop_size % 2 != 0:
                # Symmetric buffer needs to have an even number of samples
                self.pop_size += 1
            self._expl_strat = SymmParamExplStrat(self._expl_strat)

        # Dual optimization
        self.num_epoch_dual = num_epoch_dual
        self._log_eta = to.tensor([0.0], requires_grad=True)
        self.optim_mode = optim_mode.lower()
        if self.optim_mode == "scipy":
            pass
        elif self.optim_mode == "torch":
            self.optim_dual = to.optim.SGD([{"params": self._log_eta}], lr=lr_dual, momentum=0.8, weight_decay=1e-4)
            # self.optim_dual = to.optim.Adam([{'params': self._log_eta}], lr=lr_dual, eps=1e-5)  # used in [2], but unstable here
        else:
            raise pyrado.ValueErr(given=optim_mode, eq_constraint=["scipy", "torch"])

    @property
    def eta(self) -> to.Tensor:
        r""" Get the Lagrange multiplier $\eta$. In [2], $/eta$ is called $/alpha$. """
        return to.exp(self._log_eta)

    def weights(self, rets: to.Tensor) -> to.Tensor:
        """
        Compute the wights which are used to weights thy policy samples by their return.
        As stated in [2, sec 4.1], we could calculate weights using any rank preserving transformation.

        :param rets: return values per policy sample after averaging over multiple rollouts using the same policy
        :return: weights of the policy parameter samples
        """
        if self.softmax_transform:
            # Do softmax transform (softmax from PyTorch is already numerically stable)
            return to.softmax(rets / self.eta, dim=0)
        else:
            # Do numerically stabilized exp transform
            return to.exp(to.clamp((rets - to.max(rets)) / self.eta, min=-700.0))

    def dual_evaluation(
        self, eta: Union[to.Tensor, np.ndarray], rets: Union[to.Tensor, np.ndarray]
    ) -> Union[to.Tensor, np.ndarray]:
        """
        Compute the REPS dual function value for policy evaluation.

        :param eta: lagrangian multiplier (optimization variable of the dual)
        :param rets: return values per policy sample after averaging over multiple rollouts using the same policy
        :return: dual loss value
        """
        if not (
            isinstance(eta, to.Tensor)
            and isinstance(rets, to.Tensor)
            or isinstance(eta, np.ndarray)
            and isinstance(rets, np.ndarray)
        ):
            raise pyrado.TypeErr(msg="")
        return eta * self.eps + eta * logmeanexp(rets / eta)

    def dual_improvement(
        self, eta: Union[to.Tensor, np.ndarray], param_samples: to.Tensor, w: to.Tensor
    ) -> Union[to.Tensor, np.ndarray]:
        """
        Compute the REPS dual function value for policy improvement.

        :param eta: lagrangian multiplier (optimization variable of the dual)
        :param param_samples: all sampled policy parameters
        :param w: weights of the policy parameter samples
        :return: dual loss value
        """
        # The sample weights have been computed by minimizing dual_evaluation, don't track the gradient twice
        assert w.requires_grad is False

        with to.no_grad():
            distr_old = MultivariateNormal(self._policy.param_values, self._expl_strat.cov.data)

            if self.optim_mode == "scipy" and not isinstance(eta, to.Tensor):
                # We can arrive there during the 'normal' REPS routine, but also when computing the gradient (jac) for
                # the scipy optimizer. In the latter case, eta is already a tensor.
                eta = to.from_numpy(eta).to(to.get_default_dtype())
            self.wml(eta, param_samples, w)

            distr_new = MultivariateNormal(self._policy.param_values, self._expl_strat.cov.data)
            logprobs = distr_new.log_prob(param_samples)
            kl = kl_divergence(distr_new, distr_old)  # mode seeking a.k.a. exclusive KL

        if self.optim_mode == "scipy":
            loss = w.numpy() @ logprobs.numpy() + eta * (self.eps - kl.numpy())
        else:
            loss = w @ logprobs + eta * (self.eps - kl)
        return loss

    def minimize(
        self, loss_fcn: Callable, rets: to.Tensor = None, param_samples: to.Tensor = None, w: to.Tensor = None
    ):
        """
        Minimize the given dual function. Iterate `num_epoch_dual` times.

        :param loss_fcn: function to minimize, different for `wml()` and `wmap()`
        :param rets: return values per policy sample after averaging over multiple rollouts using the same policy
        :param param_samples: all sampled policy parameters
        :param w: weights of the policy parameter samples
        """
        if self.optim_mode == "scipy":
            # Use scipy optimizers
            if loss_fcn == self.dual_evaluation:
                res = optimize.minimize(
                    partial(self.dual_evaluation, rets=rets.numpy()),
                    jac=partial(get_grad_via_torch, fcn_to=partial(self.dual_evaluation, rets=rets)),
                    x0=np.array([1.0]),
                    method="SLSQP",
                    bounds=((1e-8, 1e8),),
                )
            elif loss_fcn == self.dual_improvement:
                res = optimize.minimize(
                    partial(self.dual_improvement, param_samples=param_samples, w=w),
                    jac=partial(
                        get_grad_via_torch, fcn_to=partial(self.dual_improvement, param_samples=param_samples, w=w)
                    ),
                    x0=np.array([1.0]),
                    method="SLSQP",
                    bounds=((1e-8, 1e8),),
                )
            else:
                raise pyrado.TypeErr(msg="Received an improper loss function in REPS.minimize()!")

            eta = to.from_numpy(res["x"]).to(to.get_default_dtype())
            self._log_eta = to.log(eta)

        else:
            for _ in tqdm(
                range(self.num_epoch_dual),
                total=self.num_epoch_dual,
                desc=f"Minimizing dual",
                unit="epochs",
                file=sys.stdout,
                leave=False,
            ):
                # Use PyTorch optimizers
                self.optim_dual.zero_grad()
                if loss_fcn == self.dual_evaluation:
                    loss = self.dual_evaluation(self.eta, rets)
                elif loss_fcn == self.dual_improvement:
                    loss = self.dual_improvement(self.eta, param_samples, w)
                else:
                    raise pyrado.TypeErr(msg="Received an improper loss function in REPS.minimize()!")
                loss.backward()
                self.optim_dual.step()

        if to.isnan(self._log_eta):
            raise RuntimeError(f"The dual's optimization parameter _log_eta became NaN!")

    def wml(self, eta: to.Tensor, param_samples: to.Tensor, w: to.Tensor):
        """
        Weighted maximum likelihood update of the policy's mean and the exploration strategy's covariance

        :param eta: lagrangian multiplier (optimization variable of the dual)
        :param param_samples: all sampled policy parameters
        :param w: weights of the policy parameter samples
        """
        mean_old = self._policy.param_values.clone()
        cov_old = self._expl_strat.cov.clone()

        # Update the mean
        w_sum_param_samples = to.einsum("k,kh->h", w, param_samples)
        self._policy.param_values = (eta * mean_old + w_sum_param_samples) / (to.sum(w) + eta)
        param_values_delta = self._policy.param_values - mean_old

        # Difference between all sampled policy parameters and the updated policy
        diff = param_samples - self._policy.param_values
        w_diff = to.einsum("nk,n,nh->kh", diff, w, diff)  # outer product of scaled diff, then sum over all samples

        # Update the covariance
        cov_new = (w_diff + eta * cov_old + eta * to.einsum("k,h->kh", param_values_delta, param_values_delta)) / (
            to.sum(w) + eta
        )
        self._expl_strat.adapt(cov=cov_new)

    def wmap(self, param_samples: to.Tensor, w: to.Tensor):
        """
        Weighted maximum a-posteriori likelihood update of the policy's mean and the exploration strategy's covariance

        :param param_samples: all sampled policy parameters
        :param w: weights of the policy parameter samples
        """
        # Optimize eta according to the the policy's dual function to satisfy the KL constraint
        self.minimize(self.dual_improvement, param_samples=param_samples, w=w.detach())

        # Update the policy's and exploration strategy's parameters
        self.wml(self.eta, param_samples, w.detach())

    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = param_results.mean_returns
        rets_avg_ros = to.from_numpy(rets_avg_ros).to(to.get_default_dtype())

        with to.no_grad():
            distr_old = MultivariateNormal(self._policy.param_values, self._expl_strat.cov.data)
            loss = self.dual_evaluation(self.eta, rets_avg_ros)
            self.logger.add_value("dual loss before", loss, 4)

        # Reset dual's parameter
        self._log_eta.data.fill_(0.0)

        # Optimize eta
        self.minimize(self.dual_evaluation, rets=rets_avg_ros)

        with to.no_grad():
            loss = self.dual_evaluation(self.eta, rets_avg_ros)
            self.logger.add_value("dual loss after", loss, 4)
            self.logger.add_value("eta", self.eta, 4)

        # Compute the weights using the optimized eta
        w = self.weights(rets_avg_ros)

        # Update the policy's mean and the exploration strategy's covariance
        if self.use_map:
            self.wmap(param_results.parameters, w)  # calls self.wml(param_results.parameters, w)
        else:
            self.wml(self.eta, param_results.parameters, w)

        # Logging
        distr_new = MultivariateNormal(self._policy.param_values, self._expl_strat.cov.data)
        kl_e = kl_divergence(distr_new, distr_old)  # mode seeking a.k.a. exclusive KL
        kl_i = kl_divergence(distr_old, distr_new)  # mean seeking a.k.a. inclusive KL
        self.logger.add_value("min expl strat std", to.min(self._expl_strat.std), 4)
        self.logger.add_value("avg expl strat std", to.mean(self._expl_strat.std), 4)
        self.logger.add_value("max expl strat std", to.max(self._expl_strat.std), 4)
        self.logger.add_value("expl strat entropy", self._expl_strat.get_entropy(), 4)
        self.logger.add_value("KL(new_old)", kl_e, 6)
        self.logger.add_value("KL(old_new)", kl_i, 6)
