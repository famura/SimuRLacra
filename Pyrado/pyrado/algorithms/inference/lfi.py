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
from colorama import Style, Fore
from copy import deepcopy
from tabulate import tabulate
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Callable, Type, Mapping, Tuple, List, Union, Dict

from sbi.inference import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe import PosteriorEstimator
from sbi.inference.base import simulate_for_sbi
from sbi.user_input.user_input_checks import prepare_for_sbi
from sbi.utils import posterior_nn

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.algorithms.inference.sbi_rollout_sampler import (
    SimRolloutSamplerForSBI,
    RealRolloutSamplerForSBI,
    RecRolloutSamplerForSBI,
)
from pyrado.algorithms.utils import until_thold_exceeded
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer, DomainRandWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.input_output import print_cbt


class LFI(InterruptableAlgorithm):
    """
    Learn a physically-grounded stochastic simulator using the sbi toolbox.

    TODO This class currently only works with posterior estimators and currently excludes likelihood- and density-ratio-estimators. This might be added later.
    """

    name: str = "lfi"  # TODO better acronym
    iteration_key: str = "lfi_iteration"  # logger's iteration key

    def __init__(
        self,
        save_dir: str,
        env_sim: SimEnv,
        env_real: Union[Env, str],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        prior: Distribution,
        posterior_nn_hparam: dict,
        sbi_subrtn_class: Type[PosteriorEstimator],
        summary_statistic: str,
        max_iter: int,
        num_real_rollouts: int,
        num_sim_per_real_rollout: int,
        num_eval_samples: Optional[int] = None,
        sbi_training_hparam: Optional[dict] = None,
        sbi_sampling_hparam: Optional[dict] = None,
        simulation_batch_size: Optional[int] = 1,
        use_posterior_in_the_loop: bool = True,  # TODO deprecate
        normalize_posterior: bool = True,
        subrtn_policy: Optional[Algorithm] = None,
        subrtn_policy_snapshot_mode: Optional[str] = "latest",
        thold_succ_subrtn: Optional[float] = -pyrado.inf,
        num_workers: Optional[int] = 1,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain, this can be a `RealEnv` (sim-to-real setting), a
                         `SimEnv` (sim-to-sim setting), or a directory to load a pre-recorded set of rollouts from
        :param policy: policy used for sampling the rollout, if subrtn_policy is not `None` this policy is not oly used
                       for generating the target domain rollouts, but also optimized in simulation
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param prior: distribution used by sbi as a prior
        :param posterior_nn_hparam: hyper parameters for creating the posterior's density estimator
        :param sbi_subrtn_class: sbi algorithm calls for executing the LFI, e.g. SNPE
        :param summary_statistic: the method with which the observations for LFI are computed from the rollouts
                                  Possible options:
                                 `states` (uses all observed states from rollout),
                                 `final_state` (use the last observed state from the rollout), and
                                 `bayessim` (summary statistics as proposed in [1])
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_real_rollouts: number of real-world observation received by sbi, i.e. from every rollout exactly one
                                  observation is computed
        :param num_sim_per_real_rollout: number of simulations done by sbi per real-world observation received
        :param num_eval_samples: number of samples for evaluating the posterior in `eval_posterior()`
        :param sbi_training_hparam: `dict` forwarded to sbi't `PosteriorEstimator.train()` function like
                                    `training_batch_size`, `learning_rate`, `retrain_from_scratch_each_round`, ect.
        :param sbi_sampling_hparam: keyword arguments forwarded to sbi's `DirectPosterior.sample()` function like
                                    `sample_with_mcmc`, ect.
        :param simulation_batch_size: batch size forwarded to the sbi toolbox, requires batched simulator
        :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
        :param subrtn_policy: algorithm which performs the optimization of the behavioral policy (and value-function)
        :param subrtn_policy_snapshot_mode: snapshot mode for saving during training of the subroutine
        :param thold_succ_subrtn: success threshold on the simulated system's return for the subroutine, repeat the
                                  subroutine until the threshold is exceeded or the for a given number of iterations
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env_sim, SimEnv) or isinstance(env_sim, DomainRandWrapper):
            raise pyrado.TypeErr(msg="The given env_sim must be a non-randomized simulation environment!")
        if not prior.event_shape[0] == len(dp_mapping):
            raise pyrado.ShapeErr(given=prior.event_shape, expected_match=dp_mapping)

        # Call InterruptableAlgorithm's constructor
        super().__init__(num_checkpoints=2, save_dir=save_dir, max_iter=max_iter, policy=policy, logger=logger)

        self._env_sim_sbi = env_sim  # will be randomized explicitly by sbi
        self._env_sim_trn = DomainRandWrapperBuffer(deepcopy(env_sim), randomizer=None, selection="random")
        self._env_real = env_real
        self.dp_mapping = dp_mapping
        self.summary_statistic = summary_statistic.lower()
        self.posterior_nn_hparam = posterior_nn_hparam
        self.sbi_subrtn_class = sbi_subrtn_class
        self.sbi_training_hparam = sbi_training_hparam if sbi_training_hparam is not None else dict()
        self.sbi_sampling_hparam = sbi_sampling_hparam if sbi_sampling_hparam is not None else dict()
        self.simulation_batch_size = simulation_batch_size
        self.normalize_posterior = normalize_posterior
        self.num_real_rollouts = num_real_rollouts
        self.num_sim_per_real_rollout = num_sim_per_real_rollout
        self.num_eval_samples = num_eval_samples or 10 * 2 ** len(dp_mapping)
        self.thold_succ_subrtn = float(thold_succ_subrtn)
        self.max_subrtn_rep = 3  # number of tries to exceed thold_succ_subrtn during training in simulation
        self.num_workers = num_workers

        # Temporary containers
        self._curr_observations_real = None
        self._curr_domain_param_eval = None

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

        # Initialize sbi simulator and prior
        self._sbi_simulator = None  # to be set in step()
        self._sbi_prior = None  # to be set in step()
        self._setup_sbi(prior=prior)

        # Create the algorithm instance used in sbi, e.g. SNPE-A/B/C or SNLE
        density_estimator = posterior_nn(**self.posterior_nn_hparam)  # can't be saved
        summary_writer = self.logger.printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self._sbi_subrtn = self.sbi_subrtn_class(
            prior=self._sbi_prior, density_estimator=density_estimator, summary_writer=summary_writer
        )

        # Save initial environments and the prior
        pyrado.save(self._env_sim_trn, "env_sim", "pkl", self._save_dir)
        pyrado.save(self._env_real, "env_real", "pkl", self._save_dir)
        pyrado.save(prior, "prior", "pt", self._save_dir, meta_info=None, use_state_dict=False)

    @property
    def subroutine_policy(self) -> Algorithm:
        """ Get the policy optimization subroutine. """
        return self._subrtn_policy

    @property
    def subroutine_distr(self) -> NeuralInference:
        """ Get the system identification subroutine coming from the sbi module. """
        return self._sbi_subrtn

    @property
    def sbi_simulator(self) -> Optional[Callable]:
        """ Get the simulator wrapped for sbi. """
        return self._sbi_simulator

    def _setup_sbi(self, prior: Optional[Distribution] = None, rollouts_real: Optional[List[StepSequence]] = None):
        """
        Prepare simulator and prior for usage in sbi.

        :param prior: distribution used by sbi as a prior
        :param rollouts_real: list of rollouts recorded from the real system, which are used to sync the simulations'
                              initial states
        """
        rollout_sampler = SimRolloutSamplerForSBI(
            self._env_sim_sbi, self._policy, self.dp_mapping, self.summary_statistic, rollouts_real
        )
        if prior is None:
            prior = pyrado.load(None, "prior", "pt", self._save_dir)

        # Call sbi's preparation function
        self._sbi_simulator, self._sbi_prior = prepare_for_sbi(rollout_sampler, prior)

    def step(self, snapshot_mode: str = None, meta_info: dict = None):
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        if self.curr_checkpoint == 0:
            self._curr_observations_real, _ = LFI.collect_real_observations(
                self.save_dir,
                self._env_real if not isinstance(self._env_real, str) else self._env_sim_sbi,
                self._policy,
                self.summary_statistic,
                prefix=f"iter_{self._curr_iter}",
                num_rollouts=self.num_real_rollouts,
                rec_rollouts_dir=self._env_real if isinstance(self._env_real, str) else None,
            )
            if (
                self._curr_observations_real.ndim != 2
                or self._curr_observations_real.shape[0] != self.num_real_rollouts
            ):
                raise pyrado.ShapeErr(
                    msg=f"The observations must be a 2-dim PyTorch tensor where the first dimension has as"
                    f"many entries as there are observations, but the shape is {self._curr_observations_real.shape}!"
                )

            # Initialize sbi simulator and prior, and set the initial state space
            self._setup_sbi(
                rollouts_real=pyrado.load(
                    None, "rollouts_real", "pkl", self._save_dir, meta_info=dict(prefix=f"iter_{self._curr_iter}")
                )
            )

            # First iteration
            if self._curr_iter == 0:
                # Sample parameters proposal, and simulate these parameters to obtain the observations
                domain_param, observation_sim = simulate_for_sbi(
                    simulator=self._sbi_simulator,
                    proposal=self._sbi_prior,
                    num_simulations=self.num_sim_per_real_rollout,
                    simulation_batch_size=self.simulation_batch_size,
                    num_workers=self.num_workers,
                )
                self._sbi_subrtn.append_simulations(
                    domain_param,
                    observation_sim,
                    proposal=None,  # pass None if the parameters were sampled from the prior
                )
                self._cnt_samples += self.num_sim_per_real_rollout * self._env_sim_sbi.max_steps

                # Append the first set of observations
                pyrado.save(self._curr_observations_real, "observations_real", "pt", self._save_dir)

            # Remaining training iterations
            else:
                posterior = pyrado.load(None, "posterior", "pt", self._save_dir, meta_info)

                for ro in self._curr_observations_real:
                    posterior.set_default_x(ro)

                    # Sample parameters proposal, and simulate these parameters to obtain the observations
                    domain_param, observation_sim = simulate_for_sbi(
                        simulator=self._sbi_simulator,
                        proposal=posterior,
                        num_simulations=self.num_sim_per_real_rollout,
                        simulation_batch_size=self.simulation_batch_size,
                        num_workers=self.num_workers,
                    )
                    self._sbi_subrtn.append_simulations(domain_param, observation_sim, proposal=posterior)
                    self._cnt_samples += self.num_sim_per_real_rollout * self._env_sim_sbi.max_steps

                # Append and save all observations
                prev_observations = pyrado.load(None, "observations_real", "pt", self._save_dir)
                observations_real_hist = to.cat([prev_observations, self._curr_observations_real], dim=0)
                pyrado.save(observations_real_hist, "observations_real", "pt", self._save_dir)

            self.reached_checkpoint()  # setting counter to 1

        if self.curr_checkpoint == 1:
            # Train the posterior
            self._sbi_subrtn.train(**self.sbi_training_hparam)
            posterior = self._sbi_subrtn.build_posterior(
                **self.sbi_sampling_hparam
            )  # no need to pass density_estimator, since latest is used by default
            pyrado.save(
                posterior,
                "posterior",
                "pt",
                self._save_dir,
                meta_info=dict(prefix=f"iter_{self._curr_iter}"),
                use_state_dict=False,
            )
            pyrado.save(posterior, "posterior", "pt", self._save_dir, meta_info, use_state_dict=False)

            # Logging
            self._curr_domain_param_eval, log_prob = LFI.eval_posterior(
                posterior,
                self._curr_observations_real,
                self.num_eval_samples,
                normalize_posterior=self.normalize_posterior,
            )
            self.logger.add_value("avg domain param", to.mean(self._curr_domain_param_eval, dim=[0, 1]))
            self.logger.add_value("std domain param", to.std(self._curr_domain_param_eval, dim=[0, 1]))
            self.logger.add_value("avg log prob", to.mean(log_prob))
            self.logger.add_value("num total samples", self._cnt_samples)  # here the samples are simulations

            self.reached_checkpoint()  # setting counter to 2

        if self.curr_checkpoint == 2:
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

    @staticmethod
    def collect_real_observations(
        save_dir: Optional[str],
        env: Env,
        policy: Policy,
        summary_statistic: str,
        prefix: str,
        num_rollouts: int,
        rec_rollouts_dir: Optional[str] = None,
    ) -> Tuple[to.Tensor, List[StepSequence]]:
        """
        Roll-out a (behavioral) policy on the target system for later use with the sbi module, and save the observations
        computed from the recorded rollouts.
        This method is static to facilitate evaluation of specific policies in hindsight.
        When sampling from a real environment should be substituted with loading pre-recorded rollouts, pass the
        associated simulation environment as `env`, such that some checks can be done.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param summary_statistic: the method with which the observations for LFI are computed from the rollouts.
                                  Possible options:
                                 `states` (uses all observed states from rollout),
                                 `final_state` (use the last observed state from the rollout), and
                                 `bayessim` (summary statistics as proposed in [1])
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param rec_rollouts_dir: if not `None`, load a set of pre-recorded rollouts from the given directory
        :return: 2-dim tensor of observations extracted from the rollouts, where the samples are along the first dim
        """
        if not (isinstance(inner_env(env), RealEnv) or isinstance(inner_env(env), SimEnv)):
            raise pyrado.TypeErr(given=inner_env(env), expected_type=[RealEnv, SimEnv])

        # Evaluate sequentially (necessary for sim-to-real experiments)
        if rec_rollouts_dir is not None:
            if env.name not in rec_rollouts_dir:
                print_cbt(
                    f"Are you really sure that you loaded the correct recorded rollouts? The environment's name "
                    f"{env.name} does not appear in the directory string {rec_rollouts_dir}.",
                    "r",
                )
            if str(int(1 / env.dt)) not in rec_rollouts_dir:
                print_cbt(
                    f"Are you really sure that you loaded the correct recorded rollouts? The environment's "
                    f"control frequency {1/env.dt} does not appear in the directory string {rec_rollouts_dir}.",
                    "r",
                )
            rollout_worker = RecRolloutSamplerForSBI(summary_statistic, rec_rollouts_dir, rand_init_rollout=False)
        else:
            rollout_worker = RealRolloutSamplerForSBI(env, policy, summary_statistic)

        observations_real = []
        rollouts_real = []
        for _ in tqdm(
            range(num_rollouts),
            total=num_rollouts,
            desc=Fore.CYAN + Style.BRIGHT + f"Collecting observations using {prefix}_policy" + Style.RESET_ALL,
            unit="rollouts",
            file=sys.stdout,
        ):
            observation, rollout = rollout_worker()
            observations_real.append(observation)
            rollouts_real.append(rollout)

        # Stacked to tensor, samples along 1st dimension
        observations_real = to.stack(observations_real)

        # Optionally save the data
        if save_dir is not None:
            pyrado.save(observations_real, "observations_real", "pt", save_dir, meta_info=dict(prefix=prefix))
            pyrado.save(rollouts_real, "rollouts_real", "pkl", save_dir, meta_info=dict(prefix=prefix))

        return observations_real, rollouts_real

    @staticmethod
    def eval_posterior(
        posterior: DirectPosterior,
        observations_real: to.Tensor,
        num_samples: int,
        calculate_log_probs: Optional[bool] = True,
        normalize_posterior: Optional[bool] = True,
        sbi_sampling_hparam: Optional[dict] = None,
    ) -> Tuple[to.Tensor, Optional[to.Tensor]]:
        r"""
        Evaluates the posterior by computing parameter samples given observed data, its log probability
        and the simulated trajectory.

        :param posterior: posterior to evaluate, e.g. a normalizing flow, that samples domain parameters conditioned on
                          the provided observations
        :param observations_real: observations from the real-world rollouts a.k.a. $x_o$
        :param num_samples: number of samples to draw from the posterior
        :param calculate_log_probs: if `True` the log-probabilities are computed, else `None` is returned
        :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
        :param sbi_sampling_hparam: keyword arguments forwarded to sbi's `DirectPosterior.sample()` function
        :return: domain parameters sampled form the posterior, and log-probabilities of these domain parameters
        """
        if observations_real.ndim != 2:
            raise pyrado.ShapeErr(msg="The observations must be a 2-dim PyTorch tensor!")
        num_obs, dim_obs = observations_real.shape

        # Sample domain parameters from the normalizing flow
        sbi_sampling_hparam = sbi_sampling_hparam if sbi_sampling_hparam is not None else dict()
        domain_params = to.stack(
            [posterior.sample((num_samples,), x=obs, **sbi_sampling_hparam) for obs in observations_real], dim=0
        )
        if domain_params.shape[0] != num_obs or domain_params.shape[1] != num_samples:  # shape[2] = num_domain_param
            raise pyrado.ShapeErr(given=domain_params, expected_match=(num_obs, num_samples, -1))

        # Init container
        log_prob = to.empty((num_obs, num_samples)) if calculate_log_probs else None

        if calculate_log_probs:
            # Compute the log probability
            for idx in tqdm(
                range(num_obs),
                total=num_obs,
                desc="Evaluating posterior",
                unit="observations",
                file=sys.stdout,
                leave=False,
            ):
                log_prob[idx, :] = posterior.log_prob(
                    domain_params[idx, :, :], observations_real[idx, :], normalize_posterior
                )

        return domain_params, log_prob

    @staticmethod
    def get_ml_posterior_samples(
        dp_mapping: Mapping[int, str],
        posterior: DirectPosterior,
        observations_real: to.Tensor,
        num_eval_samples: int,
        num_ml_samples: Optional[int] = 1,
        calculate_log_probs: Optional[bool] = True,
        normalize_posterior: Optional[bool] = True,
        sbi_sampling_hparam: Optional[dict] = None,
    ) -> List[List[Dict]]:
        r"""
        Evaluates the posterior and extract the `num_ml_samples` most likely domain parameter sets.

        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param posterior: posterior to evaluate, e.g. a normalizing flow, that samples domain parameters conditioned on
                          the provided observations
        :param observations_real: observations from the real-world rollouts a.k.a. $x_o$
        :param num_eval_samples: number of samples to draw from the posterior
        :param num_ml_samples: number of most likely samples, i.e. 1 equals argmax
        :param calculate_log_probs: if `True` the log-probabilities are computed, else `None` is returned
        :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
        :param sbi_sampling_hparam: keyword arguments forwarded to sbi's `DirectPosterior.sample()` function
        :return: most likely domain parameters sets sampled form the posterior
        """
        if not isinstance(num_ml_samples, int) or num_ml_samples < 1:
            raise pyrado.ValueErr(given=num_ml_samples, g_constraint="0 (int)")

        # Evaluate the posterior
        domain_params, log_probs = LFI.eval_posterior(
            posterior,
            observations_real,
            num_eval_samples,
            calculate_log_probs,
            normalize_posterior,
            sbi_sampling_hparam,
        )

        # Extract the most likely domain parameter sets for every target domain observation
        domain_params_ml = []
        for idx_r in range(domain_params.shape[0]):
            idcs_ml = to.argsort(log_probs[idx_r, :], descending=True)
            idcs_sel = idcs_ml[:num_ml_samples]
            dp_vals = domain_params[idx_r, idcs_sel, :]
            dp_vals = np.atleast_1d(dp_vals.numpy())
            domain_param_ml = [dict(zip(dp_mapping.values(), dpv)) for dpv in dp_vals]
            domain_params_ml.append(domain_param_ml)

        return domain_params_ml

    @staticmethod
    def eval_policy(
        save_dir: Optional[str],
        env: Env,
        policy: Policy,
        prefix: str,
        num_rollouts: int,
        num_workers: Optional[int] = 1,
    ) -> to.Tensor:
        """
        Evaluate a policy either in the source or in the target domain.
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_workers: number of environments for the parallel sampler (only used for SimEnv)
        :return: estimated return in the target domain
        """
        if save_dir is not None:
            print_cbt(f"Executing {prefix}_policy ...", "c", bright=True)

        if isinstance(inner_env(env), RealEnv):
            # Evaluate sequentially when evaluating on a real-world device
            rets_real = []
            for i in range(num_rollouts):
                rets_real.append(rollout(env, policy, eval=True).undiscounted_return())

        elif isinstance(inner_env(env), SimEnv):
            # Create a parallel sampler when evaluating in a simulation
            sampler = ParallelRolloutSampler(env, policy, num_workers=num_workers, min_rollouts=num_rollouts)
            ros = sampler.sample(eval=True)
            rets_real = [ro.undiscounted_return() for ro in ros]
        else:
            raise pyrado.TypeErr(given=inner_env(env), expected_type=[RealEnv, SimEnv])

        rets_real = to.as_tensor(rets_real, dtype=to.get_default_dtype())

        if save_dir is not None:
            # Save and print the evaluation results
            pyrado.save(rets_real, "returns_real", "pt", save_dir, meta_info=dict(prefix=prefix))
            print_cbt("Target domain performance", bright=True)
            print(
                tabulate(
                    [
                        ["mean return", to.mean(rets_real).item()],
                        ["std return", to.std(rets_real)],
                        ["min return", to.min(rets_real)],
                        ["max return", to.max(rets_real)],
                    ]
                )
            )

        return to.mean(rets_real)

    @staticmethod
    def fill_domain_param_buffer(env: DomainRandWrapper, dp_mapping: Mapping[int, str], domain_params: to.Tensor):
        """
        Fill the environments domain parameter buffer according to the domain parameter map, and reset the ring index.

        :param env: environment in which the domain parameters are inserted
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param domain_params: tensor of domain parameters [num_samples x dim domain param]
        """
        if not isinstance(env, DomainRandWrapperBuffer):
            raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapperBuffer)
        if domain_params.ndim != 2 or domain_params.shape[1] != len(dp_mapping):
            raise pyrado.ShapeErr(
                msg=f"The domain parameter must be a 2-dim PyTorch tensor, where the second dimension matched the "
                f"domain parameter mapping, but it has the shape {domain_params.shape}!"
            )

        domain_params = domain_params.detach().cpu().numpy()
        env.buffer = [dict(zip(dp_mapping.values(), dp)) for dp in domain_params]
        env.ring_idx = 0
        print_cbt(f"Filled the environment's buffer with {len(env.buffer)} domain parameters sets.", "g")

    def train_policy_sim(self, domain_params: to.Tensor, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param domain_params: domain parameters sampled from the posterior [shape N x D where N is the number of
                              samples and D is the number of domain parameters]
        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: estimated return of the trained policy in the target domain
        """
        if not (domain_params.ndim == 2 and domain_params.shape[1] == len(self.dp_mapping)):
            raise pyrado.ShapeErr(given=domain_params, expected_match=(-1, 2))

        # Insert the domain parameters into the wrapped environment's buffer
        LFI.fill_domain_param_buffer(self._env_sim_trn, self.dp_mapping, domain_params)

        # Set the initial state spaces of the simulation environment to match the observed initial states
        rollouts_real = pyrado.load(None, "rollouts_real", "pkl", self._save_dir, meta_info=dict(prefix=prefix))
        try:
            init_states_real = np.stack([ro.rollout_info["init_state"] for ro in rollouts_real])
        except Exception:  # TODO deprecate
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
        self._env_sim_trn.ring_idx = 0  # don't reset the buffer to eval on the same domains as trained
        avg_ret_sim = self.eval_policy(
            None, self._env_sim_trn, self._subrtn_policy.policy, prefix, self.num_eval_samples
        )
        return float(avg_ret_sim)

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._env_sim_trn, "env_sim", "pkl", self._save_dir)
            if self._subrtn_policy is None:
                # The policy is not being updated by a policy optimization subroutine
                pyrado.save(self._policy, "policy", "pt", self.save_dir, None)
            else:
                self._subrtn_policy.save_snapshot()

        else:
            raise pyrado.ValueErr(msg=f"{self.name} is not supposed be run as a subroutine!")

    def __getstate__(self):
        # Remove the unpickleable sbi-related members from this algorithm instance
        tmp_sbi_simulator = self.__dict__.pop("_sbi_simulator")
        tmp_sbi_subrtn_summary_writer = self.__dict__["_sbi_subrtn"].__dict__.pop("_summary_writer")
        tmp_sbi_subrtn_build_neural_net = self.__dict__["_sbi_subrtn"].__dict__.pop("_build_neural_net")

        # Remove the policy optimization subroutine, since it contains non-leaf tensors. These cause an error durin the
        # subsequent deepcopying
        tmp_subrtn_policy = self.__dict__.pop("_subrtn_policy", None)

        # Call Algorithm's __getstate__() without the unpickleable sbi-related members
        state_dict = super(LFI, self).__getstate__()

        # Make a deep copy of the state dict such that we can return the pickleable version and insert the sbi variables
        state_dict_copy = deepcopy(state_dict)

        # Inset them back
        self.__dict__["_sbi_simulator"] = tmp_sbi_simulator
        self.__dict__["_sbi_subrtn"]._summary_writer = tmp_sbi_subrtn_summary_writer
        self.__dict__["_sbi_subrtn"]._build_neural_net = tmp_sbi_subrtn_build_neural_net
        self.__dict__["_subrtn_policy"] = tmp_subrtn_policy

        return state_dict_copy

    def __setstate__(self, state):
        # Call Algorithm's __setstate__()
        super().__setstate__(state)

        # Reconstruct the simulator for sbi
        rollout_sampler = SimRolloutSamplerForSBI(
            self._env_sim_sbi, self._policy, self.dp_mapping, self.summary_statistic
        )
        sbi_simulator, _ = prepare_for_sbi(rollout_sampler, state["_sbi_prior"])  # sbi_prior is fine as it is
        self.__dict__["_sbi_simulator"] = sbi_simulator

        # Reconstruct the tensorboard printer with the once from this algorithm
        summary_writer = state["_logger"].printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self.__dict__["_sbi_subrtn"]._summary_writer = summary_writer

        # Set the internal sbi construction callable to None
        self.__dict__["_sbi_subrtn"]._build_neural_net = None
