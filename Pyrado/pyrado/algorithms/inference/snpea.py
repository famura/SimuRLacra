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
import torch as to
from copy import deepcopy
from sbi.inference import simulate_for_sbi
from sbi.user_input.user_input_checks import prepare_for_sbi
from torch.distributions import Distribution
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from typing import Optional

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.sampling.sbi_rollout_sampler import SimRolloutSamplerForSBI
from pyrado.logger.step import StepLogger
from pyrado.policies.special.mdn import MDNPolicy


class SNPEA(Algorithm):
    """
    Sequential Neural Posterior Estimation (SNPE-A) [1]

    [1] G. Papamakarios, I. Murray. "Fast epsilon-free inference of simulation models with Bayesian conditional
        density estimation.", NIPS, 2016
    """

    name: str = "snpea"
    iteration_key: str = "snpea_iteration"  # logger's iteration key

    def __init__(
        self,
        save_dir,
        rollout_sampler: SimRolloutSamplerForSBI,
        density_estimator: MDNPolicy,
        prior: Distribution,
        max_iter: int,
        num_sim_per_round: int,
        num_rounds: Optional[int] = 1,
        num_eval_samples: Optional[int] = 50,
        batch_size: Optional[int] = 50,
        lr: Optional[int] = 5e-4,
        max_grad_norm: Optional[int] = 5.0,
        eval_every_n_sims: Optional[int] = 10,
        simulation_batch_size: Optional[int] = 10,
        use_gaussian_proposal: Optional[bool] = False,
        num_workers: Optional[int] = 1,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param rollout_sampler: a simulator which maps domain parameter to observations
        :param density_estimator: the posterior model. Currently this works only for mixture of experts
        :param prior: a prior posterior from which the domain parameters are sampled
        :param num_sim_per_round: number of simulations done by sbi per round (i.e. iteration over the same target
                                  domain data set)
        :param num_rounds: the amount of times the algorithm is repeated, so far SNPE-A is only supporting 1 round
        :param max_iter: maximum number of trainings epochs for one round
        :param num_eval_samples: number of samples for evaluating the posterior in `eval_posterior()`
        :param batch_size: training batch size
        :param lr: learning rate
        :param max_grad_norm: clipping constant for the gradient
        :param eval_every_n_sims: evaluate the approach after a certain amount of training steps
        :param use_gaussian_proposal: flag to pretrain a proposal multi-variate normal prior distribution. If `False`
                                      the prior distribution will be used to generate domain parameters during training
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy=density_estimator, logger=logger)

        self.use_gaussian_proposal = use_gaussian_proposal
        self.num_sim_per_round = num_sim_per_round
        self.num_rounds = num_rounds
        assert num_rounds == 1  # TODO
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.num_eval_samples = num_eval_samples
        self.eval_every_n_sims = eval_every_n_sims

        self.density_estimator = density_estimator
        self.prior = prior
        self.proposal_prior = deepcopy(self.prior)
        self.rollout_sampler = rollout_sampler
        self.num_workers = num_workers
        self.simulation_batch_size = simulation_batch_size

        self._optim = to.optim.Adam([{"params": self.density_estimator.parameters()}], lr=lr, eps=1e-5)

    @property
    def posterior(self) -> MDNPolicy:
        """ Return a copy of the posterior"""
        return deepcopy(self.density_estimator)

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Get and check the data
        data_real = meta_info["rollouts_real"]
        if data_real.ndim != 2:
            pyrado.ShapeErr(given=data_real)

        # Pretrain proposal prior (see [1])
        if self.use_gaussian_proposal:
            self.pretrain_proposal(data_real)

        for idx_round in range(self.num_rounds):
            # Train the posterior
            self.train_posterior(data_real)

            # Save snapshot data
            meta_info.update(dict(round=idx_round))
            self.make_snapshot(snapshot_mode, meta_info=meta_info)

    def pretrain_proposal(self, data_real):
        """
        Pretrain a proposal prior distribution using a posterior with a single mixture component, such that the
        proposal prior remains a single gaussian.
        This is necessary if we later want to train a mixture component gaussian.

        .. note::
            This is not done in BayesSim, but strongly suggested in the SNPE-A paper [1]

        :param data_real: real-world observations
        """
        raise NotImplementedError  # TODO

    def train_posterior(self, data_real: to.Tensor):
        """
        Carry out one round of Algorithm-2 in [1]

        [1] G. Papamakarios, I. Murray. "Fast epsilon-free inference of simulation models with Bayesian conditional
            density estimation.", NIPS, 2016

        :param data_real: batch of real-world observations
        """
        num_obs = data_real.shape[0]  # TODO check for multiple real-rollouts at the same time

        # variable which tracks how many samples have been used for training
        sims_since_eval = 0
        grad_norm = []

        # Call sbi's preparation function
        sbi_simulator, sbi_prior = prepare_for_sbi(self.rollout_sampler, self.prior)

        # Generate the training data
        proposal_params, observations_sim = [], []
        for o in range(num_obs):
            if self.use_gaussian_proposal:
                self.proposal_prior.set_default_x(o)
            sim_params, sim_obs = simulate_for_sbi(
                simulator=sbi_simulator,
                proposal=sbi_prior,
                num_simulations=self.num_sim_per_round,
                simulation_batch_size=self.simulation_batch_size,
                num_workers=self.num_workers,
            )

            # sim_obs = self.rollout_sampler(sim_params)
            proposal_params.append(sim_params)
            observations_sim.append(sim_obs)
        proposal_params = to.cat(proposal_params, dim=0)
        observations_sim = to.cat(observations_sim, dim=0)

        pbar = tqdm(total=self.max_iter, desc="Training", unit="iteration", file=sys.stdout, leave=False)
        cnt_iter = 0
        while cnt_iter < self.max_iter:

            if len(proposal_params) != len(observations_sim):
                raise RuntimeError

            # Split data into batches
            for indices in BatchSampler(SubsetRandomSampler(range(proposal_params.shape[0])), self.batch_size, False):
                dp_batch = proposal_params[indices]
                obs_batch = observations_sim[indices]

                # Reset the gradients
                self._optim.zero_grad()

                # Maximizing the posterior probability
                log_probs = self.density_estimator.log_prob(dp_batch, obs_batch)
                loss = -to.mean(log_probs)
                loss.backward()

                # Clip the gradients if desired
                grad_norm.append(self.clip_grad(self.density_estimator, self.max_grad_norm))
                self._optim.step()

            pbar.update(1)
            cnt_iter += 1

    # TODO implement SNPE-A in sbi, thus use the eval_posterior method from BayesSim / NPDR
    # @staticmethod
    # @to.no_grad()
    # def eval_posterior(
    #     posterior: MDNPolicy,
    #     data_real: to.Tensor,
    #     num_samples: int,
    #     calculate_log_probs: Optional[bool] = True,
    #     disable: Optional[bool] = False,
    #     **kwargs,
    # ):
    #     r"""
    #     Evaluates the posterior by computing parameter samples given observed data, its log probability
    #     and the simulated trajectory.
    #
    #     :param posterior: posterior to evaluate, e.g. a normalizing flow, that samples domain parameters conditioned on
    #                       the provided observations
    #     :param data_real: observations from the real-world rollouts a.k.a. $x_o$
    #     :param num_samples: number of samples to draw from the posterior
    #     :param calculate_log_probs: if `True` the log-probabilities are computed, else `None` is returned
    #     :return: domain parameters sampled form the posterior, and log-probabilities of these domain parameters
    #     """
    #     if data_real.ndim != 2:
    #         raise pyrado.ShapeErr(msg="The observations must be a 2-dim PyTorch tensor!")
    #     num_obs, dim_obs = data_real.shape
    #
    #     # Sample domain parameters
    #     domain_params = to.stack([posterior.sample((num_samples,), x=x_o) for x_o in data_real], dim=0)
    #     if domain_params.shape[0] != num_obs or domain_params.shape[1] != num_samples:  # shape[2] = num_domain_param
    #         raise pyrado.ShapeErr(given=domain_params, expected_match=(num_obs, num_samples, -1))
    #
    #     # Compute the log probability if desired
    #     log_probs = to.empty((num_obs, num_samples)) if calculate_log_probs else None
    #     if calculate_log_probs:
    #         for idx in tqdm(
    #             range(num_obs),
    #             total=num_obs,
    #             desc="Evaluating posterior",
    #             unit="observations",
    #             file=sys.stdout,
    #             leave=True,
    #             disable=disable,
    #         ):
    #             log_probs[idx, :] = posterior.log_prob(domain_params[idx, :, :], data_real[idx, :])
    #
    #     return domain_params, log_probs

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.density_estimator, "posterior.pt", self._save_dir)

        else:
            # This algorithm instance is a subroutine of another algorithm
            prefix = meta_info.get("prefix", "")
            idx_round = meta_info.get("round")

            if self.num_rounds > 1:
                # Save the posterior tailored to each round
                pyrado.save(
                    self.density_estimator, "posterior.pt", self._save_dir, prefix=prefix + f"_round_{idx_round}"
                )
