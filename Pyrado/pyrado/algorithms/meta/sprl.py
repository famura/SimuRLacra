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
from typing import List

import numpy as np
import pyrado
import torch as to
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.domain_randomization.domain_parameter import SelfPacedLearnerParameter
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from scipy.optimize import NonlinearConstraint, minimize
from torch.distributions import MultivariateNormal


class MultivariateNormalWrapper:
    def __init__(self, mean: to.Tensor, cov_chol_flat: to.Tensor):
        self._mean = mean
        self._cov_chol_flat = cov_chol_flat

    @staticmethod
    def from_stacked(dim: int, stacked: np.ndarray) -> "MultivariateNormalWrapper":
        assert len(stacked.shape) == 1, "Stacked has invalid shape! Must be 1-dimensional."
        assert (
            stacked.shape[0] == 2 * dim
        ), "Stacked has invalid size! Must be 2*dim (one times for mean, a second time for covariance cholesky diagonal)."
        mean = stacked[:dim]
        cov_chol_flat = stacked[dim:]
        return MultivariateNormalWrapper(to.tensor(mean), to.tensor(cov_chol_flat))

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
        alpha_function_offset,
        alpha_function_percentage,
        discount_factor: float,
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
        super().__init__(subroutine.save_dir, subroutine.max_iter, subroutine.policy, subroutine.logger)

        self._subroutine = subroutine
        self._subroutine.save_name = "sub_algorithm"

        self._env = env

        self._kl_constraints_ub = kl_constraints_ub
        self._alpha_function_offset = alpha_function_offset
        self._alpha_function_percentage = alpha_function_percentage
        self._discount_factor = discount_factor

        spl_parameters = [
            param for param in env.randomizer.domain_params if isinstance(param, SelfPacedLearnerParameter)
        ]
        assert len(spl_parameters) == 1, "Only exactly one SPL parameter is allowed!"
        self._parameter = spl_parameters[0]

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
        self.logger.add_value(f"cur context mean for {self._parameter.name}", self._parameter.context_mean.item())
        self.logger.add_value(f"cur context cov for {self._parameter.name}", self._parameter.context_cov.item())
        dim = self._parameter.dim
        # First, train with the initial context distribution
        self._subroutine.train(snapshot_mode, self._seed, meta_info)
        # Update distribution.

        previous_distribution = MultivariateNormalWrapper(
            self._parameter.context_mean, self._parameter.context_cov_chol_flat
        )
        contexts = self._parameter.sample_buffer
        print(f"Parameter ID: {id(self._parameter)}")
        print(contexts)
        contexts_old_log_prob = self._parameter.context_distribution.log_prob(contexts)
        kl_divergence = to.distributions.kl_divergence(
            self._parameter.context_distribution, self._parameter.target_distribution
        )
        rollouts = self._subroutine.rollouts
        average_reward = np.mean(
            [ro.discounted_return(gamma=self._discount_factor) for ros in rollouts for ro in ros]
        ).item()
        values = np.asarray([ro.undiscounted_return() for ros in rollouts for ro in ros])

        def kl_constraint_fn(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(
                distribution.distribution, self._parameter.context_distribution
            )
            return kl_divergence

        def kl_constraint_fn_prime(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            kl_divergence = to.distributions.kl_divergence(
                distribution.distribution, self._parameter.context_distribution
            )
            mean_grad, cov_chol_grad = to.autograd.grad(kl_divergence, distribution.parameters())
            return np.concantenate([mean_grad.detach().numpy(), cov_chol_grad.detach().numpy()])

        def objective(x):
            distribution = MultivariateNormalWrapper.from_stacked(dim, x)
            alphas = self._calculate_alpha(self.curr_iter, average_reward, kl_divergence)
            val = self._compute_context_loss(distribution.distribution, contexts, contexts_old_log_prob, values, alphas)
            mean_grad, cov_chol_flat_grad = to.autograd.grad(val, distribution.parameters())

            return (
                -val.detach().numpy(),
                -np.concatenate([mean_grad.detach.numpy(), cov_chol_flat_grad.detach.numpy()]).astype(np.float64),
            )

        constraints = [
            NonlinearConstraint(
                fun=kl_constraint_fn,
                lb=-np.inf,
                ub=self._kl_constraints_ub,
                jac=kl_constraint_fn_prime,
                keep_feasible=True,
            )
        ]

        # noinspection PyTypeChecker
        result = minimize(
            objective,
            previous_distribution.get_stacked(),
            method="trust-constr",
            jac=True,
            constraints=constraints,
            options={"gtol": 1e-4, "xtol": 1e-6},
        )

        if result.success:
            self._parameter.adapt("context_mean", result[0])
            self._parameter.adapt("context_cov_chol_flat", result[1])
        else:
            old_f = objective(previous_distribution.get_stacked())[0]
            if kl_constraint_fn(result.x) <= self._kl_constraints_ub and result.fun < old_f:
                self._parameter.adapt("context_mean", result[0])
                self._parameter.adapt("context_cov_chol_flat", result[1])
            else:
                raise pyrado.BaseErr("Sad… :/")

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

    def _calculate_alpha(self, iteration: int, average_reward: float, kl_divergence: float):
        if iteration < self._alpha_function_offset:
            alpha = 0.0
        else:
            kl_divergence = to.clamp(kl_divergence, min=1e-10)
            average_reward = 0.0 if average_reward < 0.0 else average_reward
            alpha = to.clamp(self._alpha_function_percentage * average_reward / kl_divergence, max=1e5)
        return alpha

    def _compute_context_loss(self, distribution, contexts, contexts_old_log_prob, values, alpha):
        part1 = (to.exp(distribution.log_prob(contexts) - contexts_old_log_prob) * values).mean()
        part2 = alpha * to.distributions.kl_divergence(distribution, self._parameter.target_distribution)
        return part1 - part2
