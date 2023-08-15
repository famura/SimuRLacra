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

import os.path
from csv import DictWriter
from typing import Iterator, Optional, Tuple

import numpy as np
import torch as to
from scipy.optimize import NonlinearConstraint, minimize
from torch.distributions import MultivariateNormal

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.algorithms.utils import RolloutSavingWrapper, until_thold_exceeded
from pyrado.domain_randomization.domain_parameter import SelfPacedDomainParam
from pyrado.domain_randomization.transformations import DomainParamTransform
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence


def ravel_tril_elements(A: to.Tensor) -> to.Tensor:
    if not (len(A.shape) == 2):
        raise pyrado.ShapeErr(msg="A must be two-dimensional")
    if not (A.shape[0] == A.shape[1]):
        raise pyrado.ShapeErr(msg="A must be square")
    return to.cat([A[i, : i + 1] for i in range(A.shape[0])], dim=0)


def unravel_tril_elements(a: to.Tensor) -> to.Tensor:
    if not (len(a.shape) == 1):
        raise pyrado.ShapeErr(msg="a must be one-dimensional")
    raveled_dim = a.shape[0]
    dim = int((np.sqrt(8 * raveled_dim + 1) - 1) / 2)  # inverse Gaussian summation formula
    A = to.zeros((dim, dim)).double()
    for i in range(dim):
        A[i, : i + 1] = a[int(i * (i + 1) / 2) :][: i + 1]
    return A


class MultivariateNormalWrapper:
    """
    A wrapper for PyTorch's multivariate normal distribution with diagonal covariance.
    It is used to get a SciPy optimizer-ready version of the parameters of a distribution,
    i.e. a vector that can be used as the target variable.
    """

    def __init__(self, mean: to.Tensor, cov_chol: to.Tensor):
        """
        Constructor.

        :param mean: mean of the distribution; shape `(k,)`
        :param cov_chol: Cholesky decomposition of the covariance matrix; must be lower triangular; shape `(k, k)`
                              if it is the actual matrix or shape `(k * (k + 1) / 2,)` if it is raveled
        """
        if not (len(mean.shape) == 1):
            raise pyrado.ShapeErr(msg="mean must be one-dimensional")
        self._k = mean.shape[0]
        cov_chol_tril_is_raveled = len(cov_chol.shape) == 1
        if cov_chol_tril_is_raveled:
            if not (cov_chol.shape[0] == self._k * (self._k + 1) / 2):
                raise pyrado.ShapeErr(msg="raveled cov_chol must have shape (k (k + 1) / 2,)")
        else:
            if not (len(cov_chol.shape) == 2):
                raise pyrado.ShapeErr(msg="cov_chol must be two-dimensional")
            if not (cov_chol.shape[0] == cov_chol.shape[1]):
                raise pyrado.ShapeErr(msg="cov_chol must be square")
            if not (cov_chol.shape[0] == mean.shape[0]):
                raise pyrado.ShapeErr(msg="cov_chol and mean must have same size")
            if not (to.allclose(cov_chol, cov_chol.tril())):
                raise pyrado.ValueErr(msg="cov_chol must be lower triangular")
        self._mean = mean.clone().detach().requires_grad_(True)
        cov_chol_tril = cov_chol.clone().detach()
        if not cov_chol_tril_is_raveled:
            cov_chol_tril = ravel_tril_elements(cov_chol_tril)
        self._cov_chol_tril = cov_chol_tril.requires_grad_(True)
        self._update_distribution()

    @staticmethod
    def from_stacked(dim: int, stacked: np.ndarray) -> "MultivariateNormalWrapper":
        r"""
        Creates an instance of this class from the given stacked numpy array as generated e.g. by
        `MultivariateNormalWrapper.get_stacked(self)`.

        :param dim: dimensionality `k` of the random variable
        :param stacked: array containing the mean and standard deviations of shape `(k + k * (k + 1) / 2,)`, where the
                        first `k` entries are the mean and the last `k * (k + 1) / 2` entries are lower triangular
                        entries of the Cholesky decomposition of the covariance matrix
        :return: a `MultivariateNormalWrapper` with the given mean/cov.
        """
        if not (len(stacked.shape) == 1):
            raise pyrado.ValueErr(msg="Stacked has invalid shape! Must be 1-dimensional.")
        if not (stacked.shape[0] == dim + dim * (dim + 1) / 2):
            raise pyrado.ValueErr(given_name="stacked", msg="invalid size, must be dim + dim * (dim + 1) / 2)")

        mean = stacked[:dim]
        cov_chol_tril = stacked[dim:]

        return MultivariateNormalWrapper(to.tensor(mean).double(), to.tensor(cov_chol_tril).double())

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
        return self.cov_chol @ self.cov_chol.T

    @property
    def cov_chol(self) -> to.Tensor:
        """Get the Cholesky decomposition of the covariance; shape `(k, k)`."""
        return unravel_tril_elements(self._cov_chol_tril)

    @property
    def cov_chol_tril(self) -> to.Tensor:
        """Get the lower triangular of the Cholesky decomposition of the covariance; shape `(k * (k + 1) / 2)`."""
        return self._cov_chol_tril

    @cov_chol_tril.setter
    def cov_chol_tril(self, cov_chol_tril: to.Tensor):
        """Set the standard deviations, shape `(k,)`."""
        if not (cov_chol_tril.shape == self.cov_chol_tril.shape):
            raise pyrado.ShapeErr(given_name="cov_chol_tril", expected_match=self.cov_chol_tril.shape)

        self._cov_chol_tril = cov_chol_tril
        self._update_distribution()

    def parameters(self) -> Iterator[to.Tensor]:
        """Get the parameters (mean and lower triangular covariance Cholesky) of this distribution."""
        yield self.mean
        yield self.cov_chol_tril

    def get_stacked(self) -> np.ndarray:
        """
        Get the numpy representations of the mean and transformed covariance stacked on top of each other.

        :return: stacked mean and transformed covariance; shape `(k + k * (k + 1) / 2,)`
        """
        return np.concatenate([self.mean.detach().numpy(), self.cov_chol_tril.detach().numpy()])

    def _update_distribution(self):
        """Update `self.distribution` according to the current mean and covariance."""
        self.distribution = MultivariateNormal(self.mean, self.cov)


class SPDR(Algorithm):
    """
    Self-Paced Domain Randomization (SPDR)

    This algorithm wraps another algorithm. The main purpose is to apply self-paced RL [1].

    .. seealso::
        [1] P. Klink, H. Abdulsamad, B. Belousov, C. D'Eramo, J. Peters, and J. Pajarinen,
        "A Probabilistic Interpretation of Self-Paced Learning with Applications to Reinforcement Learning", arXiv, 2021
    """

    name: str = "spdr"

    def __init__(
        self,
        env: DomainRandWrapper,
        subroutine: Algorithm,
        kl_constraints_ub: float,
        max_iter: int,
        performance_lower_bound: float,
        var_lower_bound: Optional[float] = 0.04,
        kl_threshold: float = 0.1,
        optimize_mean: bool = True,
        optimize_cov: bool = True,
        max_subrtn_retries: int = 1,
    ):
        """
        Constructor

        :param env: environment wrapped in a DomainRandWrapper
        :param subroutine: algorithm which performs the policy/value-function optimization, which
                           must expose its sampler
        :param kl_constraints_ub: upper bound for the KL-divergence
        :param max_iter: Maximal iterations for the SPDR algorithm (not for the subroutine)
        :param performance_lower_bound: lower bound for the performance SPDR tries to stay above
                                        during distribution updates
        :param var_lower_bound: clipping value for the variance,necessary when using very small target variances; prefer
                                a log-transformation instead
        :param kl_threshold: threshold for the KL-divergence until which std_lower_bound is enforced
        :param optimize_mean: whether the mean should be changed or considered fixed
        :param optimize_cov: whether the (co-)variance should be changed or considered fixed
        :param max_subrtn_retries: how often a failed (median performance < 30 % of performance_lower_bound)
                                   training attempt of the subroutine should be reattempted
        """
        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(given_name="subroutine", given=subroutine, expected_type=Algorithm)
        if not hasattr(subroutine, "sampler"):
            raise AttributeError("The subroutine must have a sampler attribute!")
        if not typed_env(env, DomainRandWrapper):
            raise pyrado.TypeErr(given_name="env", given=env, expected_type=DomainRandWrapper)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subroutine.save_dir, max_iter, subroutine.policy, subroutine.logger)

        # Wrap the sampler of the subroutine with an rollout saving wrapper
        self._subrtn = subroutine
        self._subrtn.sampler = RolloutSavingWrapper(subroutine.sampler)
        self._subrtn.save_name = self._subrtn.name

        self._env = env

        # Properties for the variance bound and kl constraint
        self._kl_constraints_ub = kl_constraints_ub
        self._var_lower_bound = var_lower_bound
        self._kl_threshold = kl_threshold

        # Properties of the performance constraint
        self._performance_lower_bound = performance_lower_bound
        self._performance_lower_bound_reached = False

        self._optimize_mean = optimize_mean
        self._optimize_cov = optimize_cov

        self._max_subrtn_retries = max_subrtn_retries

        self._spl_parameter = None
        for param in env.randomizer.domain_params:
            if isinstance(param, SelfPacedDomainParam):
                if self._spl_parameter is None:
                    self._spl_parameter = param
                else:
                    raise pyrado.ValueErr(msg="randomizer contains more than one spl param")

        # evaluation multidim
        header = ["iteration", "objective_output", "status", "cg_stop_cond", "mean", "cov"]
        f = open(os.path.join(subroutine.save_dir, "optimizer.csv"), "w", buffering=1)
        global optimize_logger
        optimize_logger = DictWriter(f, fieldnames=header)
        optimize_logger.writeheader()

    @property
    def sample_count(self) -> int:
        # Forward to subroutine
        return self._subrtn.sample_count

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Perform a step of SPDR. This includes training the subroutine and updating the context distribution accordingly.
        For a description of the parameters see `pyrado.algorithms.base.Algorithm.step`.
        """
        self.save_snapshot()

        context_mean = self._spl_parameter.context_mean.double()
        context_cov = self._spl_parameter.context_cov.double()
        context_cov_chol = self._spl_parameter.context_cov_chol.double()
        target_mean = self._spl_parameter.target_mean.double()
        target_cov_chol = self._spl_parameter.target_cov_chol.double()

        # Add these keys to the logger as dummy values.
        self.logger.add_value("sprl number of particles", 0)
        self.logger.add_value("spdr constraint kl", 0.0)
        self.logger.add_value("spdr constraint performance", 0.0)
        self.logger.add_value("spdr objective", 0.0)
        for param_a_idx, param_a_name in enumerate(self._spl_parameter.name):
            for param_b_idx, param_b_name in enumerate(self._spl_parameter.name):
                self.logger.add_value(
                    f"context cov for {param_a_name}--{param_b_name}", context_cov[param_a_idx, param_b_idx].item()
                )
                self.logger.add_value(
                    f"context cov_chol for {param_a_name}--{param_b_name}",
                    context_cov_chol[param_a_idx, param_b_idx].item(),
                )
                if param_a_name == param_b_name:
                    self.logger.add_value(f"context mean for {param_a_name}", context_mean[param_a_idx].item())
                    break

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
        previous_distribution = MultivariateNormalWrapper(context_mean, context_cov_chol)
        target_distribution = MultivariateNormalWrapper(target_mean, target_cov_chol)

        def get_domain_param_value(ro: StepSequence, param_name: str) -> np.ndarray:
            domain_param_dict = ro.rollout_info["domain_param"]
            untransformed_param_name = param_name + DomainParamTransform.UNTRANSFORMED_DOMAIN_PARAMETER_SUFFIX
            if untransformed_param_name in domain_param_dict:
                return domain_param_dict[untransformed_param_name]
            return domain_param_dict[param_name]

        rollouts_all = self._get_sampler().rollouts
        contexts = to.tensor(
            [
                [to.from_numpy(get_domain_param_value(ro, name)) for rollouts in rollouts_all for ro in rollouts]
                for name in self._spl_parameter.name
            ],
            requires_grad=True,
        ).T

        self.logger.add_value("sprl number of particles", contexts.shape[0])

        contexts_old_log_prob = previous_distribution.distribution.log_prob(contexts.double())
        # kl_divergence = to.distributions.kl_divergence(previous_distribution.distribution, target_distribution.distribution)

        values = to.tensor([ro.undiscounted_return() for rollouts in rollouts_all for ro in rollouts])

        constraints = []

        def kl_constraint_fn(x):
            """Compute the constraint for the KL-divergence between current and proposed distribution."""
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            return kl_divergence.detach().numpy()

        def kl_constraint_fn_prime(x):
            """Compute the derivative for the KL-constraint (used for scipy optimizer)."""
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(
                previous_distribution.distribution, distribution.distribution
            )
            grads = to.autograd.grad(kl_divergence, list(distribution.parameters()))
            return np.concatenate([g.detach().numpy() for g in grads])

        constraints.append(
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                ub=self._kl_constraints_ub,
                jac=kl_constraint_fn_prime,
                # keep_feasible=True,
            )
        )

        def performance_constraint_fn(x):
            """Compute the constraint for the expected performance under the proposed distribution."""
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            return performance.detach().numpy()

        def performance_constraint_fn_prime(x):
            """Compute the derivative for the performance-constraint (used for scipy optimizer)."""
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            performance = self._compute_expected_performance(distribution, contexts, contexts_old_log_prob, values)
            grads = to.autograd.grad(performance, list(distribution.parameters()))
            return np.concatenate([g.detach().numpy() for g in grads])

        constraints.append(
            NonlinearConstraint(
                fun=performance_constraint_fn,
                lb=self._performance_lower_bound,
                ub=np.inf,
                jac=performance_constraint_fn_prime,
                # keep_feasible=True,
            )
        )

        # # Clip the bounds of the new variance either if the applied covariance transformation does not ensure
        # # non-negativity or when the KL threshold has been crossed.
        # bounds = None
        # x0, _, x0_cov_indices = previous_distribution.get_stacked()
        # if self._cov_transformation.ensures_non_negativity():
        #     lower_bound = -np.inf * np.ones_like(x0)
        #     lower_bound_is_inf = True
        # else:
        #     lower_bound = np.zeros_like(x0)
        #     lower_bound_is_inf = False
        # if self._kl_threshold != -np.inf and (self._kl_threshold < kl_divergence):
        #     if x0_cov_indices is not None and self._var_lower_bound is not None:
        #         # Further clip the x values if a standard deviation lower bound was set.
        #         lower_bound[dim:] = self._var_lower_bound
        #         lower_bound_is_inf = False
        # if not lower_bound_is_inf:
        #     # Only set the bounds if the lower bound is not negative infinity. Makes it easier for the optimizer.
        #     upper_bound = np.ones_like(x0) * np.inf
        #     bounds = Bounds(lb=lower_bound, ub=upper_bound, keep_feasible=True)
        #     x0 = np.clip(x0, bounds.lb, bounds.ub)

        # We now optimize based on the kl-divergence between target and context distribution by minimizing it
        def objective_fn(x):
            """Tries to find the minimum kl divergence between the current and the update distribution, which
            still satisfies the minimum update constraint and the performance constraint."""
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(distribution.distribution, target_distribution.distribution)
            grads = to.autograd.grad(kl_divergence, list(distribution.parameters()))

            return (
                kl_divergence.detach().numpy(),
                np.concatenate([g.detach().numpy() for g in grads]),
            )

        x0 = previous_distribution.get_stacked()

        print("Performing SPDR update.")
        try:
            # noinspection PyTypeChecker
            result = minimize(
                objective_fn,
                x0,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-4, "xtol": 1e-6},
                # bounds=bounds,
            )
            new_x = result.x
            if not result.success:
                # If optimization process was not a success
                old_f = objective_fn(previous_distribution.get_stacked())[0]
                constraints_satisfied = all((const.lb <= const.fun(result.x) <= const.ub for const in constraints))

                # std_ok = bounds is None or (np.all(bounds.lb <= result.x)) and np.all(result.x <= bounds.ub)
                std_ok = True

                update_successful = constraints_satisfied and std_ok and result.fun < old_f
                if not update_successful:
                    print(f"Update unsuccessful, keeping old SPDR parameters.")
                    new_x = x0
        except ValueError as e:
            print(f"Update failed with error, keeping old SPDR parameters.", e)
            new_x = x0

        self._adapt_parameters(dim, new_x)
        self.logger.add_value("spdr constraint kl", kl_constraint_fn(new_x).item())
        self.logger.add_value("spdr constraint performance", performance_constraint_fn(new_x).item())
        self.logger.add_value("spdr objective", objective_fn(new_x)[0].item())

    def reset(self, seed: int = None):
        # Forward to subroutine
        self._subrtn.reset(seed)
        self._get_sampler().reset_rollouts()

    def save_snapshot(self, meta_info: dict = None):
        self._get_sampler().reset_rollouts()
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            self._subrtn.save_snapshot(meta_info)

    def load_snapshot(self, parsed_args) -> Tuple[Env, Policy, dict]:
        env, policy, extra = super().load_snapshot(parsed_args)

        # Algorithm specific
        if isinstance(self._subrtn, ActorCritic):
            ex_dir = self._save_dir or getattr(parsed_args, "dir", None)
            extra["vfcn"] = pyrado.load(
                f"{parsed_args.vfcn_name}.pt", ex_dir, obj=self._subrtn.critic.vfcn, verbose=True
            )

        return env, policy, extra

    def _compute_expected_performance(
        self, distribution: MultivariateNormalWrapper, context: to.Tensor, old_log_prop: to.Tensor, values: to.Tensor
    ) -> to.Tensor:
        """Calculate the expected performance after an update step."""
        context_ratio = to.exp(distribution.distribution.log_prob(context) - old_log_prop)
        return to.mean(context_ratio * values)

    def _adapt_parameters(self, dim: int, result: np.ndarray) -> None:
        """Update the parameters of the distribution based on the result of
        the optimization step and the general algorithm settings."""
        context_distr = MultivariateNormalWrapper.from_stacked(dim, result)
        self._spl_parameter.adapt("context_mean", context_distr.mean)
        self._spl_parameter.adapt("context_cov_chol", context_distr.cov_chol)

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
            self._subrtn.init_modules(False)
        self._subrtn.reset()

        self._subrtn.train(snapshot_mode, None, meta_info)
        rollouts_all = self._get_sampler().rollouts
        return np.median([[ro.undiscounted_return() for rollouts in rollouts_all for ro in rollouts]]).item()

    def _get_sampler(self) -> RolloutSavingWrapper:
        # It is checked in the constructor that the sampler is a RolloutSavingWrapper.
        # noinspection PyTypeChecker
        return self._subrtn.sampler
