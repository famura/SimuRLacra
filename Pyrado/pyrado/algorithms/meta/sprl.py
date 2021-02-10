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
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyrado
import torch as to
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.domain_randomization.domain_parameter import SelfPacedLearnerParameter
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
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


class SPRL(Algorithm):
    """
    Self-Paced Reinforcement Leaner (SPRL)

    This algorithm wraps another algorithm. The main purpose is to apply self-paced RL (Klink et al, 2020).
    """

    name: str = "sprl"

    def __init__(
        self,
        env: DomainRandWrapper,
        subroutine: ActorCritic,
        kl_constraints_ub,
        max_iter: int,
        performance_lower_bound: float,
        std_lower_bound: float = 0.2,
        kl_threshold: float = 0.1,
    ):
        """
        Constructor

        :param env: Environment wrapped in a DomainRandWrapper.
        :param subroutine: Algorithm which performs the policy/value-function optimization.
        :param kl_constraints_ub: Upper bound for the KL-divergence
        :param alpha_function_offset: Alpha function offset
        """
        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(given=subroutine, expected_type=Algorithm)
        if not typed_env(env, DomainRandWrapper):
            raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapper)
        if not isinstance(subroutine, ActorCritic):
            raise pyrado.TypeErr(given=subroutine, expected_type=ActorCritic)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subroutine.save_dir, max_iter, subroutine.policy, subroutine.logger)

        self._subroutine = subroutine
        self._subroutine.save_name = subroutine.name

        self._env = env

        self._kl_constraints_ub = kl_constraints_ub
        self._std_lower_bound = std_lower_bound
        self._kl_threshold = kl_threshold
        self._performance_lower_bound = performance_lower_bound

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
        context_cov = to.cat([spl_param.context_cov for spl_param in self._spl_parameters]).flatten().double()

        target_mean = to.cat([spl_param.target_mean for spl_param in self._spl_parameters]).double()
        target_cov = to.cat([spl_param.target_cov for spl_param in self._spl_parameters]).flatten().double()

        # self.logger.add_value(f"cur context mean for {self._parameter.name}", self._parameter.context_mean.item())
        # self.logger.add_value(f"cur context cov for {self._parameter.name}", self._parameter.context_cov.item())
        dim = context_mean.shape[0]
        # First, train with the initial context distribution
        self._subroutine.train(snapshot_mode, self._seed, meta_info)

        # Update distribution
        previous_distribution = MultivariateNormalWrapper(context_mean, context_cov)
        target_distribution = MultivariateNormalWrapper(target_mean, target_cov)
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

        def kl_constraint_fn(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            return kl_divergence.detach().numpy()

        def kl_constraint_fn_prime(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            mean_grad, cov_chol_grad = to.autograd.grad(kl_divergence, distribution.parameters())
            return np.concatenate([mean_grad.detach().numpy(), cov_chol_grad.detach().numpy()])

        kl_constraint = NonlinearConstraint(
            fun=kl_constraint_fn,
            lb=-np.inf,
            ub=self._kl_constraints_ub,
            jac=kl_constraint_fn_prime,
            keep_feasible=True,
        )

        def performance_constraint_fn(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            return performance.detach().numpy()

        def performance_constraint_fn_prime(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            mean_grad, cov_chol_grad = to.autograd.grad(performance, distribution.parameters())
            return np.concatenate([mean_grad.detach().numpy(), cov_chol_grad.detach().numpy()])

        performance_contraint = NonlinearConstraint(
            fun=performance_constraint_fn,
            lb=self._performance_lower_bound,
            ub=np.inf,
            jac=performance_constraint_fn_prime,
            keep_feasible=True,
        )

        # optionally clip the bounds of the new variance
        if self._kl_threshold and (self._kl_threshold < kl_divergence):
            lower_bound = np.ones_like(previous_distribution.get_stacked()) * -np.inf
            lower_bound[dim] = self._std_lower_bound
            upper_bound = np.ones_like(previous_distribution.get_stacked()) * np.inf
            bounds = Bounds(lb=lower_bound, ub=upper_bound, keep_feasible=True)
            x0 = np.clip(previous_distribution.get_stacked(), lower_bound, upper_bound)
        else:
            bounds = None
            x0 = previous_distribution.get_stacked()

        objective_fn: Optional[Callable[..., Tuple[np.array, np.array]]] = None
        result = None
        constraints = None

        # check whether we are already above our performance threshold
        if performance_constraint_fn(x0) >= self._performance_lower_bound:
            self._performance_lower_bound_reached = True
            constraints = [kl_constraint, performance_contraint]

            # We now optimize based on the kl-divergence between target and context distribution by minimizing it
            def objective(x):
                distribution = MultivariateNormalWrapper.from_stacked(dim, x)
                kl_divergence = to.distributions.kl_divergence(
                    distribution.distribution, target_distribution.distribution
                )
                mean_grad, cov_chol_grad = to.autograd.grad(kl_divergence, distribution.parameters())

                return (
                    kl_divergence.detach().numpy(),
                    np.concatenate([mean_grad.detach().numpy(), cov_chol_grad.detach().numpy()]),
                )

            objective_fn = objective

        # If we have never reached the performance threshold we optimize just based on the kl constraint
        elif not self._performance_lower_bound_reached:
            constraints = [kl_constraint]

            # now we optimize on the expected performance, meaning maximizing it
            def objective(x):
                distribution = MultivariateNormalWrapper.from_stacked(dim, x)
                performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
                mean_grad, cov_chol_grad = to.autograd.grad(performance, distribution.parameters())

                return (
                    -performance.detach().numpy(),
                    -np.concatenate([mean_grad.detach().numpy(), cov_chol_grad.detach().numpy()]),
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
            mean_pointer = 0
            for param in self._spl_parameters:
                param.adapt("context_mean", to.tensor(result.x[mean_pointer:mean_pointer+param.dim]))
                mean_pointer += param.dim
                param.adapt("context_cov_chol_flat", to.tensor(result.x[mean_pointer:mean_pointer+param.dim]))
                mean_pointer += param.dim
        # we have a result but the optimization process was not a success
        elif result:
            old_f = objective_fn(previous_distribution.get_stacked())[0]
            constraints_satisfied = all((const.lb <= const.fun(result.x) <= const.ub for const in constraints))

            std_ok = bounds is None or (np.all(bounds.lb <= result.x)) and np.all(result.x <= bounds.ub)

            if constraints_satisfied and std_ok and result.fun < old_f:
                mean_pointer = 0
                for param in self._spl_parameters:
                    param.adapt("context_mean", to.tensor(result.x[mean_pointer:mean_pointer+param.dim]))
                    mean_pointer += param.dim
                    param.adapt("context_cov_chol_flat", to.tensor(result.x[mean_pointer:mean_pointer+param.dim]))
                    mean_pointer += param.dim
            else:
                print(f"Update unsuccessfull, keeping old values spl parameters")

        # Reset environment.
        self._subroutine.reset()
        self._env.reset()

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
