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
from typing import Union

import torch as to
from sbi.inference import SNPE_C
from sbi.inference.base import simulate_for_sbi

import pyrado
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.utils.input_output import print_cbt


class BayesSim(SBIBase):
    """
    BayesSim [1] using Sequential Neural Posterior Estimation (SNPE-A) [3]

    For convenience, we provide the possibility to train a policy of the last posterior, i.e. after the inference was
    done. This is not part of the original BayesSim algorithm [1]. Note that Online BayesSim [2], apart from using a
    different embedding, alternates between training the policy or the internal model of the policy (see the last
    paragraph of Section V.B.), thus has a significantly different data generation procedure.

    .. seealso::
        [1] F. Ramos, R.C. Possas, D. Fox, "BayesSim: adaptive domain randomization via probabilistic inference for
            robotics simulators", RSS, 2019
        [2] R.C. Possas, L. Barcelos, R. Oliveira, D. Fox, F. Ramos, "Online BayesSim for Combined Simulator Parameter
            Inference and Policy Improvement", IROS, 2020
        [3] G. Papamakarios, I. Murray. "Fast epsilon-free inference of simulation models with Bayesian conditional
            density estimation.", NIPS, 2016
    """

    name = "bayessim"
    iteration_key = "bayessim_iteration"

    def __init__(self, env_sim: Union[SimEnv, EnvWrapper], policy: Policy, downsampling_factor: int = 1, **kwargs):
        """
        Constructor

        :param env_sim: randomized simulation environment a.k.a. source domain
        :param policy: policy used for sampling the rollouts in the target domain at the beginning of each iteration.
                       If `subrtn_policy` is not `None` this policy is also trained at the very last iteration.
        :param downsampling_factor: downsampling factor for the embedding which is used for pre-processing the data
                                    before passing it to the posterior, 1 means no downsampling
        :param kwargs: forwarded the superclass constructor
        """
        # Construct the same embedding as in [1]
        embedding = BayesSimEmbedding(
            spec=env_sim.spec,
            dim_data=RolloutSamplerForSBI.get_dim_data(env_sim.spec),
            downsampling_factor=downsampling_factor,
            use_cuda=policy.device != "cpu",
        )

        # Call SBIBase's constructor
        super().__init__(
            env_sim=env_sim,
            policy=policy,
            subrtn_sbi_class=SNPE_C,  # TODO replace by SNPE-A when available
            embedding=embedding,
            num_checkpoints=3,
            init_checkpoint=0,
            max_iter=1,  # BayesSim only runs SNPE-A (could be multi-round) once on the initially collected trajectories
            use_rec_act=True,  # BayesSim requires the trajectories to be recorded beforehand
            **kwargs,
        )

    def step(self, snapshot_mode: str = "latest", meta_info: dict = None):
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        if self.curr_checkpoint == 0:
            # Check if the rollout files already exist
            if osp.isfile(osp.join(self._save_dir, "data_real.pt")) and osp.isfile(
                osp.join(self._save_dir, "rollouts_real.pkl")
            ):
                # Rollout files do exist (can be when continuing a previous experiment)
                self._curr_data_real = pyrado.load("data_real.pt", self._save_dir)
                print_cbt(f"Loaded existing rollout data.", "w")

            else:
                # Rollout files do not exist yet (usual case)
                self._curr_data_real, _ = SBIBase.collect_data_real(
                    self.save_dir,
                    self._env_real,
                    self._policy,
                    self._embedding,
                    num_rollouts=self.num_real_rollouts,
                    num_segments=self.num_segments,
                    len_segments=self.len_segments,
                )

                # Save the target domain data
                pyrado.save(self._curr_data_real, "data_real.pt", self._save_dir)

            # Initialize sbi simulator and prior
            self._setup_sbi(
                prior=self._sbi_prior,
                rollouts_real=pyrado.load("rollouts_real.pkl", self._save_dir),
            )

            self.reached_checkpoint()  # setting counter to 1

        if self.curr_checkpoint == 1:
            # The proposal of the first round is the prior
            proposal = self._sbi_prior

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
                        prefix=f"iter_{self._curr_iter}",  # for compatibility with load_posterior()
                    )

                if self.num_sbi_rounds > 1:
                    # Save the posterior tailored to each round
                    pyrado.save(
                        posterior,
                        "posterior.pt",
                        self._save_dir,
                        prefix=f"iter_{self._curr_iter}_round_{idx_r}",  # for compatibility with load_posterior()
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
                wrapped_trn_fcn(self._curr_domain_param_eval.squeeze())

            self.reached_checkpoint()  # setting counter to 0

        # Save snapshot data
        self.make_snapshot(snapshot_mode, None, meta_info)
