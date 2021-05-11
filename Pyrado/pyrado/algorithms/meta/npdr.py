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

import os.path as osp
from typing import Mapping, Optional, Type, Union

import torch as to
from sbi.inference.base import simulate_for_sbi
from sbi.inference.snpe import PosteriorEstimator
from torch.distributions import Distribution

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.sbi_embeddings import Embedding
from pyrado.utils.input_output import print_cbt


class NPDR(SBIBase):
    """Neural Posterior Domain Randomization (NPDR)"""

    name: str = "npdr"
    iteration_key: str = "npdr_iteration"  # logger's iteration key

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env_sim: Union[SimEnv, EnvWrapper],
        env_real: Union[Env, str],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        prior: Distribution,
        subrtn_sbi_class: Type[PosteriorEstimator],
        embedding: Embedding,
        max_iter: int,
        num_real_rollouts: int,
        num_sim_per_round: int,
        num_segments: int = None,
        len_segments: int = None,
        use_rec_act: bool = True,
        num_sbi_rounds: int = 1,
        num_eval_samples: Optional[int] = None,
        posterior_hparam: Optional[dict] = None,
        subrtn_sbi_training_hparam: Optional[dict] = None,
        subrtn_sbi_sampling_hparam: Optional[dict] = None,
        simulation_batch_size: int = 1,
        normalize_posterior: bool = True,
        subrtn_policy: Optional[Algorithm] = None,
        subrtn_policy_snapshot_mode: str = "latest",
        thold_succ_subrtn: float = -pyrado.inf,
        num_workers: int = 4,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain, this can be a `RealEnv` (sim-to-real setting), a
                         `SimEnv` (sim-to-sim setting), or a directory to load a pre-recorded set of rollouts from
        :param policy: policy used for sampling the rollouts at the beginning of each iteration. If `subrtn_policy` is
                       not `None` this policy is also trained after the inference step in every NPDR iteration.
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param prior: distribution used by sbi as a prior
        :param subrtn_sbi_class: sbi algorithm calls for executing the LFI, e.g. SNPE
        :param embedding: embedding used for pre-processing the data before passing it to the posterior
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_real_rollouts: number of real-world rollouts received by sbi, i.e. from every rollout exactly one
                                  data set is computed
        :param num_sim_per_round: number of simulations done by sbi per round (i.e. iteration over the same target
                                  domain data set)
        :param num_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        :param use_rec_act: if `True` the recorded actions form the target domain are used to generate the rollout
                            during simulation (feed-forward). If `False` there policy is used to generate (potentially)
                            state-dependent actions (feed-back).
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param num_sbi_rounds: set to an integer > 1 to use multi-round sbi. This way the posteriors (saved as
                               `..._round_NUMBER...` will be tailored to the data of that round, where `NUMBER`
                               counts up each round (modulo `num_real_rollouts`). If `num_sbi_rounds` = 1, the posterior
                               is called amortized (it has never seen any target domain data).
        :param num_eval_samples: number of samples for evaluating the posterior in `eval_posterior()`
        :param posterior_hparam: hyper parameters for creating the posterior's density estimator
        :param subrtn_sbi_training_hparam: dict forwarded to sbi's `PosteriorEstimator.train()` function like
                                           `training_batch_size`, `learning_rate`, `retrain_from_scratch_each_round`, ect.
        :param subrtn_sbi_sampling_hparam: keyword arguments forwarded to sbi's `DirectPosterior.sample()` function like
                                          `sample_with_mcmc`, ect.
        :param simulation_batch_size: batch size forwarded to the sbi toolbox, requires batched simulator
        :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
        :param subrtn_policy: algorithm which performs the optimization of the behavioral policy (and value-function)
        :param subrtn_policy_snapshot_mode: snapshot mode for saving during policy optimization
        :param thold_succ_subrtn: success threshold on the simulated system's return for the subroutine, repeat the
                                  subroutine until the threshold is exceeded or the for a given number of iterations
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call SBIBase's constructor
        super().__init__(
            num_checkpoints=3,
            init_checkpoint=-1,
            save_dir=save_dir,
            env_sim=env_sim,
            env_real=env_real,
            policy=policy,
            dp_mapping=dp_mapping,
            prior=prior,
            subrtn_sbi_class=subrtn_sbi_class,
            embedding=embedding,
            max_iter=max_iter,
            num_real_rollouts=num_real_rollouts,
            num_sim_per_round=num_sim_per_round,
            num_segments=num_segments,
            len_segments=len_segments,
            use_rec_act=use_rec_act,
            num_sbi_rounds=num_sbi_rounds,
            num_eval_samples=num_eval_samples,
            posterior_hparam=posterior_hparam,
            subrtn_sbi_training_hparam=subrtn_sbi_training_hparam,
            subrtn_sbi_sampling_hparam=subrtn_sbi_sampling_hparam,
            simulation_batch_size=simulation_batch_size,
            normalize_posterior=normalize_posterior,
            subrtn_policy=subrtn_policy,
            subrtn_policy_snapshot_mode=subrtn_policy_snapshot_mode,
            thold_succ_subrtn=thold_succ_subrtn,
            num_workers=num_workers,
            logger=logger,
        )

    def step(self, snapshot_mode: str = "latest", meta_info: dict = None):
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        if self.curr_checkpoint == -1:
            if self._subrtn_policy is not None:
                # Train the behavioral policy using the nominal domain parameters
                self._subrtn_policy.train(snapshot_mode=self._subrtn_policy_snapshot_mode)  # overrides policy.pt
            self.reached_checkpoint()  # setting counter to 0

        if self.curr_checkpoint == 0:
            # Check if the rollout files already exist
            if (
                osp.isfile(osp.join(self._save_dir, f"iter_{self.curr_iter}_data_real.pt"))
                and osp.isfile(osp.join(self._save_dir, "data_real.pt"))
                and osp.isfile(osp.join(self._save_dir, "rollouts_real.pkl"))
            ):
                # Rollout files do exist (can be when continuing a previous experiment)
                self._curr_data_real = pyrado.load("data_real.pt", self._save_dir, prefix=f"iter_{self.curr_iter}")
                print_cbt(f"Loaded existing rollout data for iteration {self.curr_iter}.", "w")

            else:
                # Rollout files do not exist yet (usual case)
                self._curr_data_real, _ = SBIBase.collect_data_real(
                    self.save_dir,
                    self._env_real,
                    self._policy,
                    self._embedding,
                    prefix=f"iter_{self._curr_iter}",
                    num_rollouts=self.num_real_rollouts,
                    num_segments=self.num_segments,
                    len_segments=self.len_segments,
                )

                # Save the target domain data
                if self._curr_iter == 0:
                    # Append the first set of data
                    pyrado.save(self._curr_data_real, "data_real.pt", self._save_dir)
                else:
                    # Append and save all data
                    prev_data = pyrado.load("data_real.pt", self._save_dir)
                    data_real_hist = to.cat([prev_data, self._curr_data_real], dim=0)
                    pyrado.save(data_real_hist, "data_real.pt", self._save_dir)

            # Initialize sbi simulator and prior
            self._setup_sbi(
                prior=self._sbi_prior,
                rollouts_real=pyrado.load("rollouts_real.pkl", self._save_dir, prefix=f"iter_{self._curr_iter}"),
            )

            self.reached_checkpoint()  # setting counter to 1

        if self.curr_checkpoint == 1:
            # Load the latest proposal, this can be the prior or the amortized posterior of the last iteration
            proposal = self.get_latest_unconditioned_proposal()

            # Multi-round sbi
            for idx_r in range(self.num_sbi_rounds):
                # Sample parameters proposal, and simulate these parameters to obtain the data
                domain_param, data_sim = simulate_for_sbi(
                    simulator=self._sbi_simulator,
                    proposal=proposal,
                    num_simulations=self.num_sim_per_round,
                    simulation_batch_size=self.simulation_batch_size,
                    num_workers=self.num_workers,
                )
                self._cnt_samples += self.num_sim_per_round * self._env_sim_sbi.max_steps

                # Append simulations and proposals for sbi
                self._subrtn_sbi.append_simulations(
                    domain_param,
                    data_sim,
                    proposal=proposal,  # do not pass proposal arg for SNLE or SNRE
                )

                # Train the posterior
                density_estimator = self._subrtn_sbi.train(**self.subrtn_sbi_training_hparam)
                posterior = self._subrtn_sbi.build_posterior(density_estimator, **self.subrtn_sbi_sampling_hparam)

                # Save the posterior of this iteration before tailoring it to the data (when it is still amortized)
                if idx_r == 0:
                    pyrado.save(
                        posterior,
                        "posterior.pt",
                        self._save_dir,
                        prefix=f"iter_{self._curr_iter}",
                    )

                if self.num_sbi_rounds > 1:
                    # Save the posterior tailored to each round
                    pyrado.save(
                        posterior,
                        "posterior.pt",
                        self._save_dir,
                        prefix=f"iter_{self._curr_iter}_round_{idx_r}",
                    )

                    # Set proposal of the next round to focus on the next data set.
                    # set_default_x() expects dim [1, num_rollouts * data_samples]
                    proposal = posterior.set_default_x(self._curr_data_real)

                # Override the latest posterior
                pyrado.save(posterior, "posterior.pt", self._save_dir)

            self.reached_checkpoint()  # setting counter to 2

        if self.curr_checkpoint == 2:
            # Logging (the evaluation can be time-intensive)
            posterior = pyrado.load("posterior.pt", self._save_dir)
            self._curr_domain_param_eval, log_probs = SBIBase.eval_posterior(
                posterior,
                self._curr_data_real,
                self.num_eval_samples,
                normalize_posterior=self.normalize_posterior,
            )
            self.logger.add_value(  # max likelihood domain parameter set
                "ml domain param",
                to.mean(self._curr_domain_param_eval[:, to.argmax(log_probs, dim=1), :], dim=[0, 1]),
                2,
            )
            self.logger.add_value("std domain param", to.std(self._curr_domain_param_eval, dim=[0, 1]), 2)
            self.logger.add_value("avg log prob", to.mean(log_probs), 4)
            self.logger.add_value("num total samples", self._cnt_samples)  # here the samples are simulations

            self.reached_checkpoint()  # setting counter to 3

        if self.curr_checkpoint == 3:
            # Policy optimization
            if self._subrtn_policy is not None:
                # Train the behavioral policy using the posterior samples obtained before, repeat if the resulting
                # policy did not exceed the success threshold
                wrapped_trn_fcn = until_thold_exceeded(self.thold_succ_subrtn, self.max_subrtn_rep)(
                    self.train_policy_sim
                )
                wrapped_trn_fcn(self._curr_domain_param_eval.squeeze(), prefix=f"iter_{self._curr_iter}")

            self.reached_checkpoint()  # setting counter to 0

        # Save snapshot data
        self.make_snapshot(snapshot_mode, None, meta_info)
