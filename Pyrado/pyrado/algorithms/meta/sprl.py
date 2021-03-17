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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pyrado
import torch as to
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.domain_randomization.domain_parameter import SelfPacedLearnerParameter
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.algorithms.utils import until_thold_exceeded
from scipy.optimize import NonlinearConstraint, minimize, Bounds
from torch import distributions
from torch.distributions import MultivariateNormal


class MultivariateNormalWrapper:
    def __init__(self, mean: to.Tensor, cov_chol_flat: to.Tensor):
        self._mean = mean.clone().detach().requires_grad_(True)
        self._cov_chol_flat = cov_chol_flat.clone().detach().requires_grad_(True)
        self.distribution = MultivariateNormal(self.mean, self.cov)

    @staticmethod
    def from_stacked(dim: int, stacked: np.ndarray) -> "MultivariateNormalWrapper":
        assert len(stacked.shape) == 1, "Stacked has invalid shape! Must be 1-dimensional."
        assert (
            stacked.shape[0] == 2 * dim
        ), "Stacked has invalid size! Must be 2*dim (one times for mean, a second time for covariance cholesky diagonal)."
        mean = stacked[:dim]
        cov_chol_flat = stacked[dim:]
        return MultivariateNormalWrapper(to.tensor(mean).double(), to.tensor(cov_chol_flat).double())

    @property
    def dim(self):
        return self._mean.shape[0]

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean: to.Tensor):
        assert mean.shape == self.mean.shape, "New mean shape differs from current mean shape!"
        self._mean = mean
        self._update_distribution()

    @property
    def cov(self):
        return (self._cov_chol_flat ** 2).diag()

    @property
    def cov_chol_flat(self):
        return self._cov_chol_flat

    @cov_chol_flat.setter
    def cov_chol_flat(self, cov_chol_flat: to.Tensor):
        assert (
            cov_chol_flat.shape == self.cov_chol_flat.shape
        ), "New cov chol flat shape differs from current mean shape!"
        self._cov_chol_flat = cov_chol_flat
        self._update_distribution()

    def parameters(self) -> List[to.Tensor]:
        return [self.mean, self.cov_chol_flat]

    def get_stacked(self) -> np.ndarray:
        return np.concatenate([self.mean.detach().numpy(), self.cov_chol_flat.detach().numpy()])

    def _update_distribution(self):
        self.distribution = MultivariateNormal(self.mean, self.cov)


class ParameterAgnosticMultivariateNormalWrapper(MultivariateNormalWrapper):
    def __init__(self, mean: to.Tensor, cov_chol_flat: to.Tensor, mean_is_parameter: bool, cov_is_parameter: bool):
        super().__init__(mean, cov_chol_flat)

        self._mean_is_parameter = mean_is_parameter
        self._cov_is_parameter = cov_is_parameter

    def from_stacked(self, stacked: np.ndarray) -> "ParameterAgnosticMultivariateNormalWrapper":
        assert len(stacked.shape) == 1, "Stacked has invalid shape! Must be 1-dimensional."
        expected_dim_multiplier = 0
        if self._mean_is_parameter:
            expected_dim_multiplier += 1
        if self._cov_is_parameter:
            expected_dim_multiplier += 1
        assert (
            stacked.shape[0] == expected_dim_multiplier * self.dim
        ), f"Stacked has invalid size! Must be {expected_dim_multiplier}*dim."

        if self._mean_is_parameter and self._cov_is_parameter:
            mean = stacked[: self.dim]
            cov_chol_flat = stacked[self.dim :]
        elif self._mean_is_parameter and not self._cov_is_parameter:
            mean = stacked[: self.dim]
            cov_chol_flat = self.cov_chol_flat
        elif not self._mean_is_parameter and self._cov_is_parameter:
            mean = self.mean
            cov_chol_flat = stacked
        else:
            mean = self.mean
            cov_chol_flat = self.cov_chol_flat

        if type(mean) == np.ndarray:
            mean = to.tensor(mean).double()

        if type(cov_chol_flat) == np.ndarray:
            cov_chol_flat = to.tensor(cov_chol_flat).double()

        return ParameterAgnosticMultivariateNormalWrapper(
            mean=mean,
            cov_chol_flat=cov_chol_flat,
            mean_is_parameter=self._mean_is_parameter,
            cov_is_parameter=self._cov_is_parameter,
        )

    def parameters(
        self, return_mean_cov_indices: bool = False
    ) -> Union[List[to.Tensor], Tuple[List[to.Tensor], Optional[List[int]], Optional[List[int]]]]:
        params = []
        if self._mean_is_parameter:
            params.append(self.mean)
        if self._cov_is_parameter:
            params.append(self.cov_chol_flat)
        if return_mean_cov_indices:
            pointer = 0
            mean_indices, cov_indices = None, None
            if self._mean_is_parameter:
                mean_indices = list(range(pointer, pointer + self.dim))
                pointer += self.dim
            if self._cov_is_parameter:
                cov_indices = list(range(pointer, pointer + self.dim))
                pointer += self.dim
            return params, mean_indices, cov_indices
        return params

    def get_stacked(
        self, return_mean_cov_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[List[int]], Optional[List[int]]]]:
        parameters = self.parameters(return_mean_cov_indices=return_mean_cov_indices)
        if return_mean_cov_indices:
            parameters, mean_indices, cov_indices = parameters
        else:
            mean_indices, cov_indices = None, None
        stacked = np.concatenate([p.detach().numpy() for p in parameters])
        if return_mean_cov_indices:
            return stacked, mean_indices, cov_indices
        return stacked


class SPRL(Algorithm):
    """
    Self-Paced Reinforcement Leaner (SPRL)

    This algorithm wraps another algorithm. The main purpose is to apply self-paced RL (Klink et al, 2020).
    """

    name: str = "sprl"

    def __init__(
        self,
        env: DomainRandWrapper,
        subroutine: Algorithm,
        kl_constraints_ub,
        max_iter: int,
        performance_lower_bound: float,
        std_lower_bound: float = 0.2,
        kl_threshold: float = 0.1,
        optimize_mean: bool = True,
        optimize_cov: bool = True,
        max_subrtn_retries: int = 1,
    ):
        """
        Constructor

        :param env: Environment wrapped in a DomainRandWrapper.
        :param subroutine: Algorithm which performs the policy/value-function optimization.
        :param kl_constraints_ub: Upper bound for the KL-divergence
        :param max_subrtn_retries: How often a failed (median performance < 30 % of performance_lower_bound) training attempt of the subroutine should be reattempted
        """

        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(given=subroutine, expected_type=Algorithm)
        if not typed_env(env, DomainRandWrapper):
            raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapper)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subroutine.save_dir, max_iter, subroutine.policy, subroutine.logger)

        self._subroutine = subroutine
        self._subroutine.save_name = subroutine.name

        self._env = env

        self._kl_constraints_ub = kl_constraints_ub
        self._std_lower_bound = std_lower_bound
        self._kl_threshold = kl_threshold
        self._performance_lower_bound = performance_lower_bound
        self._optimize_mean = optimize_mean
        self._optimize_cov = optimize_cov
        self._max_subrtn_retries = max_subrtn_retries

        self._performance_lower_bound_reached = False

        self._spl_parameters = [
            param for param in env.randomizer.domain_params if isinstance(param, SelfPacedLearnerParameter)
        ]

        self._seed = None

    @property
    def sub_algorithm(self) -> Algorithm:
        """ Get the policy optimization subroutine. """
        return self._subroutine

    @property
    def sample_count(self) -> int:
        return self._subroutine.sample_count

    def train(self, snapshot_mode: str = "latest", seed: int = None, meta_info: dict = None):
        self._seed = seed
        super().train(snapshot_mode, seed, meta_info)

    def step(self, snapshot_mode: str, meta_info: dict = None):
        self.save_snapshot()

        context_mean = to.cat([spl_param.context_mean for spl_param in self._spl_parameters]).double()
        context_cov_chol = to.cat([spl_param.context_cov_chol_flat for spl_param in self._spl_parameters]).double()

        target_mean = to.cat([spl_param.target_mean for spl_param in self._spl_parameters]).double()
        target_cov_chol = to.cat([spl_param.target_cov_chol_flat for spl_param in self._spl_parameters]).double()

        for param in self._spl_parameters:
            self.logger.add_value(f"cur context mean for {param.name}", param.context_mean.item())
            self.logger.add_value(f"cur context cov for {param.name}", param.context_cov.item())

        dim = context_mean.shape[0]

        # if we are in the first iteration and have a bad performance
        # we want to completely reset the policy if training is unsuccessful
        reset_policy = False
        if self.curr_iter == 0:
            reset_policy = True
        until_thold_exceeded(self._performance_lower_bound * 0.3, self._max_subrtn_retries)(
            self._train_subroutine_and_evaluate_perf
        )(snapshot_mode, meta_info, reset_policy)

        # Update distribution
        previous_distribution = ParameterAgnosticMultivariateNormalWrapper(
            context_mean, context_cov_chol, self._optimize_mean, self._optimize_cov
        )
        target_distribution = ParameterAgnosticMultivariateNormalWrapper(
            target_mean, target_cov_chol, self._optimize_mean, self._optimize_cov
        )
        rollouts = self._subroutine.rollouts
        contexts = to.tensor(
            np.array(
                [
                    [stepseq.rollout_info["domain_param"][param.name] for rollout in rollouts for stepseq in rollout]
                    for param in self._spl_parameters
                ]
            ),
            requires_grad=True,
        ).T

        contexts_old_log_prob = previous_distribution.distribution.log_prob(contexts.double())
        kl_divergence = to.distributions.kl_divergence(
            previous_distribution.distribution, target_distribution.distribution
        )

        values = to.tensor([ro.undiscounted_return() for ros in rollouts for ro in ros])

        values = to.tensor([ro.undiscounted_return() for ros in rollouts for ro in ros])

        values = to.tensor([ro.undiscounted_return() for ros in rollouts for ro in ros])

        def kl_constraint_fn(x):
            distribution = previous_distribution.from_stacked(x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            return kl_divergence.detach().numpy()

        def kl_constraint_fn_prime(x):
            distribution = previous_distribution.from_stacked(x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            grads = to.autograd.grad(kl_divergence, distribution.parameters())
            return np.concatenate([g.detach().numpy() for g in grads])

        kl_constraint = NonlinearConstraint(
            fun=kl_constraint_fn,
            lb=-np.inf,
            ub=self._kl_constraints_ub,
            jac=kl_constraint_fn_prime,
            # keep_feasible=True,
        )

        def performance_constraint_fn(x):
            distribution = previous_distribution.from_stacked(x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            return performance.detach().numpy()

        def performance_constraint_fn_prime(x):
            distribution = previous_distribution.from_stacked(x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            grads = to.autograd.grad(performance, distribution.parameters())
            return np.concatenate([g.detach().numpy() for g in grads])

        performance_contraint = NonlinearConstraint(
            fun=performance_constraint_fn,
            lb=self._performance_lower_bound,
            ub=np.inf,
            jac=performance_constraint_fn_prime,
            # keep_feasible=True,
        )

        # optionally clip the bounds of the new variance
        bounds = None
        x0, _, x0_cov_indices = previous_distribution.get_stacked(return_mean_cov_indices=True)
        if self._kl_threshold != -np.inf and (self._kl_threshold < kl_divergence):
            lower_bound = np.ones_like(x0) * -np.inf
            if x0_cov_indices is not None:
                lower_bound[x0_cov_indices] = self._std_lower_bound
            upper_bound = np.ones_like(x0) * np.inf
            # bounds = Bounds(lb=lower_bound, ub=upper_bound, keep_feasible=True)
            bounds = Bounds(lb=lower_bound, ub=upper_bound)
            x0 = np.clip(x0, lower_bound, upper_bound)

        objective_fn: Optional[Callable[..., Tuple[np.array, np.array]]] = None
        result = None
        constraints = None

        # check whether we are already above our performance threshold
        if performance_constraint_fn(x0) >= self._performance_lower_bound:
            self._performance_lower_bound_reached = True
            constraints = [kl_constraint, performance_contraint]

            # We now optimize based on the kl-divergence between target and context distribution by minimizing it
            def objective(x):
                distribution = previous_distribution.from_stacked(x)
                kl_divergence = to.distributions.kl_divergence(
                    distribution.distribution, target_distribution.distribution
                )
                grads = to.autograd.grad(kl_divergence, distribution.parameters())

                return (
                    kl_divergence.detach().numpy(),
                    np.concatenate([g.detach().numpy() for g in grads]),
                )

            objective_fn = objective

        # If we have never reached the performance threshold we optimize just based on the kl constraint
        elif not self._performance_lower_bound_reached:
            constraints = [kl_constraint]

            # now we optimize on the expected performance, meaning maximizing it
            def objective(x):
                distribution = previous_distribution.from_stacked(x)
                performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
                grads = to.autograd.grad(performance, distribution.parameters())

                return (
                    -performance.detach().numpy(),
                    -np.concatenate([g.detach().numpy() for g in grads]),
                )

            objective_fn = objective

        if objective_fn:
            result = minimize(
                objective_fn,
                x0,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-4, "xtol": 1e-6},
                bounds=bounds,
            )
        if result and result.success:
            self._adapt_parameters(result.x)
        # we have a result but the optimization process was not a success
        elif result:
            old_f = objective_fn(previous_distribution.get_stacked())[0]
            constraints_satisfied = all((const.lb <= const.fun(result.x) <= const.ub for const in constraints))

            std_ok = bounds is None or (np.all(bounds.lb <= result.x)) and np.all(result.x <= bounds.ub)

            if constraints_satisfied and std_ok and result.fun < old_f:
                self._adapt_parameters(result.x)
            else:
                print(f"Update unsuccessful, keeping old values spl parameters")

    def reset(self, seed: int = None):
        # Forward to subroutine
        self._subroutine.reset(seed)

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            self._subroutine.save_snapshot(meta_info)

    # This is the same operation as we do in part1 of the previous computation of the context loss
    # But since we want to use it as a constraint, we have to calculate it here explicitly
    def _compute_expected_performance(
        self, distribution: MultivariateNormalWrapper, context: to.Tensor, old_log_prop: to.Tensor, values: to.Tensor
    ) -> to.Tensor:
        context_ratio = to.exp(distribution.distribution.log_prob(context) - old_log_prop)
        return to.mean(context_ratio * values)

    def _adapt_parameters(self, result: np.array) -> None:
        for i, param in enumerate(self._spl_parameters):
            if self._optimize_mean:
                param.adapt("context_mean", to.tensor(result[i : i + param.dim]))
            if self._optimize_cov and self._optimize_mean:
                pointer = i + param.dim * len(self._spl_parameters)
                param.adapt("context_cov_chol_flat", to.tensor(result[pointer : pointer + param.dim]))
            elif self._optimize_cov:
                param.adapt("context_cov_chol_flat", to.tensor(result[i : i + param.dim]))

    def _train_subroutine_and_evaluate_perf(
        self, snapshot_mode: str, meta_info: dict = None, reset_policy: bool = False
    ) -> float:
        if reset_policy:
            self._subroutine.policy.reset()
        self._subroutine.reset()

        self._subroutine.train(snapshot_mode, self._seed, meta_info)
        rollouts = self._subroutine.rollouts
        x = np.median([[ro.undiscounted_return() for ros in rollouts for ro in ros]])
        return x
