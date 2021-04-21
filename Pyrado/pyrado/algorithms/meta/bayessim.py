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

from copy import deepcopy
from typing import Mapping, Optional, Union

import torch as to
from torch.distributions import Distribution

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.snpea import SNPEA
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper, DomainRandWrapperBuffer
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.special.mdn import MDNPolicy
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding
from pyrado.sampling.sbi_rollout_sampler import SimRolloutSamplerForSBI
from pyrado.spaces.box import InfBoxSpace
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.data_types import EnvSpec, merge_dicts


class BayesSim(Algorithm):
    """
    BayesSim [1] using Sequential Neural Posterior Estimation (SNPE-A) [2]

    .. seealso::
        [1] F. Ramos, R.C. Possas, D. Fox, "BayesSim: adaptive domain randomization via probabilistic inference for
            robotics simulators", RSS, 2019
        [2] G. Papamakarios, I. Murray. "Fast epsilon-free inference of simulation models with Bayesian conditional
            density estimation.", NIPS, 2016
    """

    name = "bayessim"
    iteration_key = "bayessim_iteration"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env_sim: SimEnv,
        env_real: Union[Env, str],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        prior: Distribution,
        embedding: BayesSimEmbedding,
        max_iter: int,
        num_real_rollouts: int,
        num_sim_per_round: int,
        num_segments: int = None,
        len_segments: int = None,
        num_sbi_rounds: int = 1,
        num_eval_samples: Optional[int] = None,
        posterior_hparam: Optional[dict] = None,
        subrtn_sbi_training_hparam: Optional[dict] = None,
        subrtn_sbi_snapshot_mode: Optional[str] = "latest",
        subrtn_policy: Optional[Algorithm] = None,
        subrtn_policy_snapshot_mode: str = "latest",
        thold_succ_subrtn: float = -pyrado.inf,
        num_workers: Optional[int] = 1,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        .. note::
            The `subrtn_sbi` ist not from the sbi package, it is called this way because it also does sbi.

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain, this can be a `RealEnv` (sim-to-real setting), a
                         `SimEnv` (sim-to-sim setting), or a directory to load a pre-recorded set of rollouts from
        :param policy: policy used for sampling the rollout, if subrtn_policy is not `None` this policy is not oly used
                       for generating the target domain rollouts, but also optimized in simulation
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param prior: distribution used by sbi as a prior
        :param embedding: embedding used for pre-processing the data before passing it to the posterior
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_real_rollouts: number of real-world rollouts received by sbi, i.e. from every rollout exactly one
                                  data set is computed
        :param num_sim_per_round: number of simulations done by sbi per round (i.e. iteration over the same target domain data set)
        :param num_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param num_eval_samples: number of samples for evaluating the posterior in `eval_posterior()`
        :param posterior_hparam: parameters for creating the posterior of type `MDNPolicy`, e.g. `dict(num_comp=5)` to
                                 specify the number of mixture components. For more options see `MDNPolicy`.
        :param subrtn_sbi_training_hparam: hyper-parameters for training the `MDNPolicy` posterior
        :param subrtn_policy: algorithm which performs the optimization of the behavioral policy (and value-function)
        :param subrtn_sbi_snapshot_mode: snapshot mode for saving during inference
        :param subrtn_policy_snapshot_mode: snapshot mode for saving during policy optimization
        :param thold_succ_subrtn: success threshold on the simulated system's return for the subroutine, repeat the
                                  subroutine until the threshold is exceeded or the for a given number of iterations
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env_sim, SimEnv) or isinstance(env_sim, DomainRandWrapper):
            raise pyrado.TypeErr(msg="The given env_sim must be a non-randomized simulation environment!")
        if not prior.event_shape[0] == len(dp_mapping):
            raise pyrado.ShapeErr(given=prior.event_shape, expected_match=dp_mapping)
        if posterior_hparam is None:
            posterior_hparam = dict()
        elif not isinstance(posterior_hparam, dict):
            raise pyrado.TypeErr(given=posterior_hparam, expected_type=dict)
        if subrtn_sbi_training_hparam is None:
            subrtn_sbi_training_hparam = dict()
        elif not isinstance(subrtn_sbi_training_hparam, dict):
            raise pyrado.TypeErr(given=subrtn_sbi_training_hparam, expected_type=dict)

        # Call Algorithm's constructor
        super(BayesSim, self).__init__(save_dir=save_dir, max_iter=max_iter, policy=policy, logger=logger)

        # Set defaults which can be overwritten
        subrtn_sbi_training_hparam = merge_dicts([dict(max_iter=200), subrtn_sbi_training_hparam])
        posterior_hparam = merge_dicts([dict(num_comp=5), posterior_hparam])

        self._env_sim = env_sim
        self._env_sim_trn = DomainRandWrapperBuffer(deepcopy(env_sim), randomizer=None, selection="random")
        self._env_real = env_real
        self.dp_mapping = dp_mapping
        self._embedding = embedding
        self.num_sim_per_round = num_sim_per_round
        self.num_real_rollouts = num_real_rollouts
        self.num_segments = num_segments
        self.len_segments = len_segments
        self.num_sbi_rounds = num_sbi_rounds
        self.num_eval_samples = num_eval_samples or 10 * 2 ** len(dp_mapping)
        self.thold_succ_subrtn = float(thold_succ_subrtn)
        self.max_subrtn_rep = 3  # number of tries to exceed thold_succ_subrtn during training in simulation
        self.num_workers = num_workers

        # Create a rollout sampler
        self.rollout_sampler = SimRolloutSamplerForSBI(
            self._env_sim,
            self._policy,
            self.dp_mapping,
            self._embedding,
            self.num_segments,
            self.len_segments,
        )

        # Create the posterior
        moe_spec = EnvSpec(
            obs_space=InfBoxSpace(self.embedding.dim_output),
            act_space=InfBoxSpace(len(dp_mapping)),
        )
        density_estimator = MDNPolicy(spec=moe_spec, **posterior_hparam)

        # Temporary containers
        self._curr_data_real = None
        self._curr_domain_param_eval = None

        # System identification subroutine
        self._subrtn_sbi = SNPEA(
            save_dir,
            self.rollout_sampler,
            density_estimator,
            prior,
            num_sim_per_round=num_sim_per_round,
            num_rounds=num_sbi_rounds,
            **subrtn_sbi_training_hparam,
        )
        self._subrtn_sbi.save_name = "subrtn_distr"
        self._subrtn_sbi_snapshot_mode = subrtn_sbi_snapshot_mode

        # Optional policy optimization subroutine
        self._subrtn_policy = subrtn_policy
        if isinstance(self._subrtn_policy, Algorithm):
            self._subrtn_policy_snapshot_mode = subrtn_policy_snapshot_mode
            self._subrtn_policy.save_name = "subrtn_policy"
            # Check that the behavioral policy is the one that is being updated
            if self._subrtn_policy.policy is not self.policy:
                raise pyrado.ValueErr(
                    msg="The policy is the policy subroutine is not the same as the one used by "
                    "the system identification (sbi) subroutine!"
                )

        # Save initial environments and the prior
        pyrado.save(self._env_sim, "env_sim.pkl", self._save_dir)
        pyrado.save(self._env_real, "env_real.pkl", self._save_dir)
        pyrado.save(embedding, "embedding.pt", self._save_dir)
        pyrado.save(prior, "prior.pt", self._save_dir)

    @property
    def subroutine_policy(self) -> Algorithm:
        """ Get the policy optimization subroutine. """
        return self._subrtn_policy

    @property
    def subroutine_distr(self) -> SNPEA:
        """ Get the system identification subroutine coming from the sbi module. """
        return self._subrtn_sbi

    @property
    def embedding(self) -> BayesSimEmbedding:
        """ Get the embedding used to compute the features from the rollouts. """
        return self._embedding

    @property
    def posterior(self) -> MDNPolicy:
        """ Get the current (conditional) posterior density estimator. """
        return self._subrtn_sbi.posterior

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

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

        # Reset the inference subroutine
        self._subrtn_sbi.reset()

        # Train the posterior, and save the current iteration's as well as round's posterior
        self._subrtn_sbi.train(
            snapshot_mode=self._subrtn_sbi_snapshot_mode,
            meta_info=dict(rollouts_real=self._curr_data_real, prefix=f"iter_{self._curr_iter}"),
        )

        # Override the latest posterior
        pyrado.save(self._subrtn_sbi.posterior, "posterior.pt", self._save_dir)

        # TODO comment this in once SNPE-A is implemented like in the sbi package
        # Logging (the evaluation can be time-intensive)
        # posterior = pyrado.load("posterior.pt", self._save_dir, meta_info)
        # self._curr_domain_param_eval, log_probs = SBIBase.eval_posterior(
        #     posterior,
        #     self._curr_data_real,
        #     self.num_eval_samples,
        #     normalize_posterior=False,
        # )
        # self.logger.add_value(  # max likelihood domain parameter set
        #     "ml domain param",
        #     to.mean(self._curr_domain_param_eval[:, to.argmax(log_probs, dim=1), :], dim=[0, 1]),
        #     2,
        # )
        # self.logger.add_value("std domain param", to.std(self._curr_domain_param_eval, dim=[0, 1]), 2)
        # self.logger.add_value("avg log prob", to.mean(log_probs), 4)
        # self.logger.add_value("num total samples", self._cnt_samples)  # here the samples are simulations

        # Policy optimization
        if self._subrtn_policy is not None:
            # Train the behavioral policy using the posterior samples obtained before, repeat if the resulting
            # policy did not exceed the success threshold
            wrapped_trn_fcn = until_thold_exceeded(self.thold_succ_subrtn, self.max_subrtn_rep)(self.train_policy_sim)
            wrapped_trn_fcn(self._curr_domain_param_eval.squeeze(), prefix=f"iter_{self._curr_iter}")

        # Save snapshot data
        self.make_snapshot(snapshot_mode, None, meta_info)

    def train_policy_sim(self, domain_params: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param domain_params: domain parameters sampled from the posterior [shape N x D where N is the number of
                              samples and D is the number of domain parameters]
        :param prefix: set a prefix to the saved file name, use "" for no prefix
        :return: estimated return of the trained policy in the target domain
        """
        if not (domain_params.ndim == 2 and domain_params.shape[1] == len(self.dp_mapping)):
            raise pyrado.ShapeErr(given=domain_params, expected_match=(-1, 2))

        # Insert the domain parameters into the wrapped environment's buffer
        NPDR.fill_domain_param_buffer(self._env_sim_trn, self.dp_mapping, domain_params)

        # Set the initial state spaces of the simulation environment to match the observed initial states
        rollouts_real = pyrado.load("rollouts_real.pkl", self._save_dir, prefix=prefix)
        init_states_real = np.stack([ro.states[0, :] for ro in rollouts_real])
        if not init_states_real.shape == (len(rollouts_real), self._env_sim_trn.state_space.flat_dim):
            raise pyrado.ShapeErr(
                given=init_states_real, expected_match=(len(rollouts_real), self._env_sim_trn.state_space.flat_dim)
            )
        self._env_sim_trn.wrapped_env.init_space = DiscreteSpace(init_states_real)

        # Reset the subroutine algorithm which includes resetting the exploration
        self._cnt_samples += self._subrtn_policy.sample_count
        self._subrtn_policy.reset()

        # Propagate the updated training environment to the SamplerPool's workers
        if hasattr(self._subrtn_policy, "sampler"):
            self._subrtn_policy.sampler.reinit(env=self._env_sim_trn)
        else:
            raise pyrado.KeyErr(keys="sampler", container=self._subrtn_policy)

        # Train a policy in simulation using the subroutine
        self._subrtn_policy.train(snapshot_mode=self._subrtn_policy_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        assert len(self._env_sim_trn.buffer) == self.num_eval_samples
        self._env_sim_trn.ring_idx = 0  # don't reset the buffer to eval on the same domains as trained
        avg_ret_sim = SBIBase.eval_policy(
            None, self._env_sim_trn, self._subrtn_policy.policy, prefix, self.num_eval_samples
        )
        return float(avg_ret_sim)

    def save_snapshot(self, meta_info: dict = None):
        super(BayesSim, self).save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            if self._subrtn_policy is None:
                # The policy is not being updated by a policy optimization subroutine
                pyrado.save(self._policy, "policy.pt", self.save_dir, use_state_dict=True)
            else:
                self._subrtn_policy.save_snapshot()
