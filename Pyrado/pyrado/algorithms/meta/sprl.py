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

from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as to
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from torch.distributions import MultivariateNormal

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import RolloutSavingWrapper, until_thold_exceeded
from pyrado.domain_randomization.domain_parameter import SelfPacedDomainParam
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.sampling.expose_sampler import ExposedSampler


class MultivariateNormalWrapper:
    """
    A wrapper for PyTorch's multivariate normal distribution with diagonal covariance.
    It is used to get a SciPy optimizer-ready version of the parameters of a distribution,
    i.e. a vector that can be used as the target variable.
    """

    def __init__(self, mean: to.Tensor, cov_chol_flat: to.Tensor):
        """
        Constructor.

        :param mean: mean of the distribution; shape `(k,)`
        :param cov_chol_flat: standard deviations of the distribution; shape `(k,)`
        """
        self._mean = mean.clone().detach().requires_grad_(True)
        self._cov_chol_flat = cov_chol_flat.clone().detach().requires_grad_(True)
        self.distribution = MultivariateNormal(self.mean, self.cov)

    @staticmethod
    def from_stacked(dim: int, stacked: np.ndarray) -> "MultivariateNormalWrapper":
        r"""
        Creates an instance of this class from the given stacked numpy array as generated e.g. by
        `MultivariateNormalWrapper.get_stacked(self)`.

        :param dim: dimensionality `k` of the random variable
        :param stacked: array containing the mean and standard deviations of shape `(2 * k,)`, where the first `k`
                        entries are the mean and the last `k` entries are the standard deviations
        :return: a `MultivariateNormalWrapper` with the given mean/cov.
        """
        if not (len(stacked.shape) == 1):
            raise pyrado.ValueErr(msg="Stacked has invalid shape! Must be 1-dimensional.")
        if not (stacked.shape[0] == 2 * dim):
            raise pyrado.ValueErr(
                msg="Stacked has invalid size!"
                "Must be 2*dim (one times for mean, a second time for covariance cholesky diagonal)."
            )

        mean = stacked[:dim]
        cov_chol_flat = stacked[dim:]

        return MultivariateNormalWrapper(to.tensor(mean).double(), to.tensor(cov_chol_flat).double())

    @property
    def dim(self):
        """Get the size (dimensionality) of the random variable."""
        return self._mean.shape[0]

    @property
    def mean(self):
        """Get the mean."""
        return self._mean

    @mean.setter
    def mean(self, mean: to.Tensor):
        """Set the mean."""
        if not (mean.shape == self.mean.shape):
            raise pyrado.ShapeErr(given_name="mean", expected_match=self.mean.shape)

        self._mean = mean
        self._update_distribution()

    @property
    def cov(self):
        """Get the covariance matrix, shape `(k, k)`."""
        return (self._cov_chol_flat ** 2).diag()

    @property
    def cov_chol_flat(self):
        """
        Get the standard deviations, i.e. the diagonal entries of the Cholesky decomposition of the covariance matrix,
        shape `(k,)`.
        """
        return self._cov_chol_flat

    @cov_chol_flat.setter
    def cov_chol_flat(self, cov_chol_flat: to.Tensor):
        """Set the standard deviations, shape `(k,)`."""
        if not (cov_chol_flat.shape == self.cov_chol_flat.shape):
            raise pyrado.ShapeErr(given_name="cov_chol_flat", expected_match=self.cov_chol_flat.shape)

        self._cov_chol_flat = cov_chol_flat
        self._update_distribution()

    def parameters(self) -> Generator[to.Tensor, None, None]:
        """Get the parameters (mean and standard deviations) of this distribution."""
        yield self.mean
        yield self.cov_chol_flat

    def get_stacked(self) -> np.ndarray:
        """
        Get the numpy representations of the mean and standard deviations stacked on top of each other.

        :return: stacked mean and standard deviations; shape `(k,)`
        """
        return np.concatenate([self.mean.detach().numpy(), self.cov_chol_flat.detach().numpy()])

    def _update_distribution(self):
        """Update `self.distribution` according to the current mean and covariance."""
        self.distribution = MultivariateNormal(self.mean, self.cov)


class ParameterAgnosticMultivariateNormalWrapper(MultivariateNormalWrapper):
    """
    Version of the `MultivariateNormalWrapper` that is able to exclude either the mean of the covariance from the
    parameters and the stacking. This can be readily used for optimizing the mean or covariance while keeping the
    other fixed.
    """

    def __init__(self, mean: to.Tensor, cov_chol_flat: to.Tensor, mean_is_parameter: bool, cov_is_parameter: bool):
        """
        Constructor.

        :param mean: mean of the distribution; shape `(k,)`
        :param cov_chol_flat: standard deviations of the distribution; shape `(k,)`
        :param mean_is_parameter: if `True`, the mean is treated as a parameter and returned from `get_stacked`
                                  and similar methods.
        :param cov_is_parameter: if `True`, the covariance is treated as a parameter and returned from `get_stacked`
                                 and similar methods.
        """
        super().__init__(mean, cov_chol_flat)

        self._mean_is_parameter = mean_is_parameter
        self._cov_is_parameter = cov_is_parameter

    def from_stacked(self, stacked: np.ndarray) -> "ParameterAgnosticMultivariateNormalWrapper":
        """
        Builds a new `ParameterAgnosticMultivariateNormalWrapper` from the given stacked values. In contrast to
        `MultivariateNormalWrapper.from_stacked(dim, stacked)`, this does not require a dimensionality as it is an
        instance rather than a static method. Also, the stacked representations has to either contain the mean or the
        standard deviations or both, according the the values originally passed to the constructor. If one of them is
        not treated as a parameter, the current values is copied instead.

        :param stacked: the stacked representation of the parameters according to the documentation above; can have
                        either shape `(0,)`, `(k,)`, or `(2 * k)`
        :return: a `ParameterAgnosticMultivariateNormalWrapper` with the new values for the parameters
        """
        if not (len(stacked.shape) == 1):
            raise pyrado.ValueErr(msg="Stacked has invalid shape! Must be 1-dimensional.")

        expected_dim_multiplier = 0
        if self._mean_is_parameter:
            expected_dim_multiplier += 1
        if self._cov_is_parameter:
            expected_dim_multiplier += 1
        if not (stacked.shape[0] == expected_dim_multiplier * self.dim):
            raise pyrado.ValueErr(msg=f"Stacked has invalid size! Must be {expected_dim_multiplier}*dim.")

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

    def parameters(self) -> Generator[to.Tensor, None, None]:
        """Get the list of parameters according to the values passed to the constructor."""
        if self._mean_is_parameter:
            yield self.mean
        if self._cov_is_parameter:
            yield self.cov_chol_flat

    def get_stacked(
        self, return_mean_cov_indices: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[List[int]], Optional[List[int]]]]:
        """
        Like `MultivariateNormalWrapper.get_stacked(self)`, returns a numpy array with the mean and standard deviations
        stacked on top of each other if the respective value is a parameter. Additionally, if `return_mean_cov_indices`
        is `True`, also the indices of the mean/cov values are returned which can be used for addressing them.

        :param return_mean_cov_indices: if `True`, additionally to just the parameters also the indices are returned
        :return: if `return_mean_cov_indices` is `True`, a tuple `(stacked, mean_indices, cov_indices)`, where the
                 latter two might be `None` if the respective value is not a parameter; if `return_mean_cov_indices`
                 is `False`, just the stacked values; the stacked values can have shape `(0,)`, `(k,)`, or `(2 * k,)`,
                 depending on if none, only mean/cov, or both are parameters, respectively
        """
        parameters = self.parameters()
        stacked = np.concatenate([p.detach().numpy() for p in parameters])

        if return_mean_cov_indices:
            pointer = 0
            mean_indices, cov_indices = None, None
            if self._mean_is_parameter:
                mean_indices = list(range(pointer, pointer + self.dim))
                pointer += self.dim
            if self._cov_is_parameter:
                cov_indices = list(range(pointer, pointer + self.dim))
                pointer += self.dim
            return stacked, mean_indices, cov_indices

        return stacked


class SPRL(Algorithm):
    """
    Self-Paced Reinforcement Leaner (SPRL)

    This algorithm wraps another algorithm. The main purpose is to apply self-paced RL [1].

    .. seealso::
        [1] P. Klink, H. Abdulsamad, B. Belousov, C. D'Eramo, J. Peters, and J. Pajarinen,
        "A Probabilistic Interpretation of Self-Paced Learning with Applications to Reinforcement Learning", arXiv, 2021
    """

    name: str = "sprl"

    def __init__(
        self,
        env: DomainRandWrapper,
        subroutine: Algorithm,
        kl_constraints_ub: float,
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

        :param env: environment wrapped in a DomainRandWrapper
        :param subroutine: algorithm which performs the policy/value-function optimization, which
                           must inherit from `ExposedSampler`
        :param kl_constraints_ub: upper bound for the KL-divergence
        :param max_iter: Maximal iterations for the SPRL algorithm (not for the subroutine)
        :param performance_lower_bound: lower bound for the performance SPRL tries to stay above
                                        during distribution updates
        :param std_lower_bound: clipping value for the standard deviation,necessary when using
                                         very small target variances
        :param kl_threshold: threshold for the KL-divergence until which std_lower_bound is enforced
        :param optimize_mean: whether the mean should be changed or considered fixed
        :param optimize_cov: whether the (co-)variance should be changed or considered fixed
        :param max_subrtn_retries: how often a failed (median performance < 30 % of performance_lower_bound)
                                   training attempt of the subroutine should be reattempted
        """
        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(
                given=subroutine,
                expected_type=Algorithm,
                msg="Subroutine must inherit from *Algorithm* and ExposedSampler!",
            )
        if not isinstance(subroutine, ExposedSampler):
            raise pyrado.TypeErr(
                given=subroutine,
                expected_type=ExposedSampler,
                msg="Subroutine must inherit from Algorithm and *ExposedSampler*!",
            )
        if not typed_env(env, DomainRandWrapper):
            raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapper)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subroutine.save_dir, max_iter, subroutine.policy, subroutine.logger)

        # Wrap the sampler of the subroutine with an rollout saving wrapper
        ros = RolloutSavingWrapper(subroutine.sampler)
        subroutine.sampler = ros

        # Using a Union here is not really correct, but it makes PyCharm's type hinting work
        # suggest properties from both Algorithm and ExposedSampler
        self._subroutine: Union[Algorithm, ExposedSampler] = subroutine
        self._subroutine.save_name = self._subroutine.name

        self._env = env

        # Properties for the variance bound and kl constraint
        self._kl_constraints_ub = kl_constraints_ub
        self._std_lower_bound = std_lower_bound
        self._kl_threshold = kl_threshold

        # Properties of the performance constraint
        self._performance_lower_bound = performance_lower_bound
        self._performance_lower_bound_reached = False

        self._optimize_mean = optimize_mean
        self._optimize_cov = optimize_cov

        self._max_subrtn_retries = max_subrtn_retries

        self._spl_parameters = [
            param for param in env.randomizer.domain_params if isinstance(param, SelfPacedDomainParam)
        ]

    @property
    def sub_algorithm(self) -> Algorithm:
        """Get the policy optimization subroutine."""
        return self._subroutine

    @property
    def sample_count(self) -> int:
        # Forward to subroutine
        return self._subroutine.sample_count

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Perform a step of SPRL. This includes training the subroutine and updating the context distribution accordingly.
        For a description of the parameters see `pyrado.algorithms.base.Algorithm.step`.
        """
        self.save_snapshot()

        context_mean = to.cat([spl_param.context_mean for spl_param in self._spl_parameters]).double()
        context_cov_chol = to.cat([spl_param.context_cov_chol_flat for spl_param in self._spl_parameters]).double()

        target_mean = to.cat([spl_param.target_mean for spl_param in self._spl_parameters]).double()
        target_cov_chol = to.cat([spl_param.target_cov_chol_flat for spl_param in self._spl_parameters]).double()

        for param in self._spl_parameters:
            self.logger.add_value(f"cur context mean for {param.name}", param.context_mean.item())
            self.logger.add_value(f"cur context cov for {param.name}", param.context_cov.item())

        dim = context_mean.shape[0]

        # If we are in the first iteration and have a bad performance,
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

        rollouts_all = self._subroutine.sampler.rollouts
        contexts = to.tensor(
            [
                [
                    to.from_numpy(ro.rollout_info["domain_param"][param.name])
                    for rollouts in rollouts_all
                    for ro in rollouts
                ]
                for param in self._spl_parameters
            ],
            requires_grad=True,
        ).T

        contexts_old_log_prob = previous_distribution.distribution.log_prob(contexts.double())
        kl_divergence = to.distributions.kl_divergence(
            previous_distribution.distribution, target_distribution.distribution
        )

        values = to.tensor([ro.undiscounted_return() for rollouts in rollouts_all for ro in rollouts])

        def kl_constraint_fn(x):
            """Compute the constraint for the KL-divergence between current and proposed distribution."""
            distribution = previous_distribution.from_stacked(x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            return kl_divergence.detach().numpy()

        def kl_constraint_fn_prime(x):
            """Compute the derivative for the KL-constraint (used for scipy optimizer)."""
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
            """Compute the constraint for the expected performance under the proposed distribution."""
            distribution = previous_distribution.from_stacked(x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            return performance.detach().numpy()

        def performance_constraint_fn_prime(x):
            """Compute the derivative for the performance-constraint (used for scipy optimizer)."""
            distribution = previous_distribution.from_stacked(x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            grads = to.autograd.grad(performance, distribution.parameters())
            return np.concatenate([g.detach().numpy() for g in grads])

        performance_constraint = NonlinearConstraint(
            fun=performance_constraint_fn,
            lb=self._performance_lower_bound,
            ub=np.inf,
            jac=performance_constraint_fn_prime,
            # keep_feasible=True,
        )

        # Optionally clip the bounds of the new variance
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

        # Check whether we are already above our performance threshold
        if performance_constraint_fn(x0) >= self._performance_lower_bound:
            self._performance_lower_bound_reached = True
            constraints = [kl_constraint, performance_constraint]

            # We now optimize based on the kl-divergence between target and context distribution by minimizing it
            def objective(x):
                """Optimization objective before the minimum specified performance was reached.
                Tries to find the minimum kl divergence between the current and the update distribution, which
                still satisfies the minimum update constraint and the performance constraint."""
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

            # Now we optimize on the expected performance, meaning maximizing it
            def objective(x):
                """Optimization objective when the minimum specified performance was reached.
                Tries to maximizes performance while still satisfying the minimum kl update constraint."""
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

        # We have a result but the optimization process was not a success
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
        self._subroutine.sampler.reset_rollouts()

    def save_snapshot(self, meta_info: dict = None):
        self._subroutine.sampler.reset_rollouts()
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            self._subroutine.save_snapshot(meta_info)

    def _compute_expected_performance(
        self, distribution: MultivariateNormalWrapper, context: to.Tensor, old_log_prop: to.Tensor, values: to.Tensor
    ) -> to.Tensor:
        """Calculate the expected performance after an update step."""
        context_ratio = to.exp(distribution.distribution.log_prob(context) - old_log_prop)
        return to.mean(context_ratio * values)

    def _adapt_parameters(self, result: np.array) -> None:
        """Update the parameters of the distribution based on the result of
        the optimization step and the general algorithm settings."""
        for i, param in enumerate(self._spl_parameters):
            if self._optimize_mean:
                param.adapt("context_mean", to.tensor(result[i : i + param.dim]))
            if self._optimize_cov and self._optimize_mean:
                pointer = i + param.dim * len(self._spl_parameters)
                param.adapt("context_cov_chol_flat", to.tensor(result[pointer : pointer + param.dim]))
            elif self._optimize_cov:
                param.adapt("context_cov_chol_flat", to.tensor(result[i : i + param.dim]))

    def _train_subroutine_and_evaluate_perf(
        self, snapshot_mode: str, meta_info: dict = None, reset_policy: bool = False, **kwargs
    ) -> float:
        """
        Internal method required by the `until_thold_exceeded` function.
        The parameters are the same as for the regular `train()` call and are explained there.

        :param reset_policy: if `True` the policy will be reset before training
        :return: the median undiscounted return
        """
        if reset_policy:
            self._subroutine.init_modules(False)
        self._subroutine.reset()

        self._subroutine.train(snapshot_mode, None, meta_info)
        rollouts_all = self._subroutine.sampler.rollouts
        x = np.median([[ro.undiscounted_return() for rollouts in rollouts_all for ro in rollouts]])
        return x
