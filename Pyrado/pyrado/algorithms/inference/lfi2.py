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
from colorama import Style, Fore
from copy import deepcopy
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Optional, Callable, Type, Union, Mapping, Tuple

from sbi.inference import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe import PosteriorEstimator
from sbi.inference.base import simulate_for_sbi
from sbi.user_input.user_input_checks import prepare_for_sbi
from sbi.utils import posterior_nn

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.algorithms.inference.sbi_rollout_sampler import SimRolloutSamplerForSBI, RealRolloutSamplerForSBI
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy


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
        env_real: Union[RealEnv, EnvWrapper, SimEnv],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        prior: Distribution,
        posterior_nn_hparam: dict,
        sbi_subrtn_class: Type[PosteriorEstimator],
        summary_statistic: str,
        max_iter: int,
        num_real_rollouts: int,
        num_sim_per_real_rollout: int,
        num_eval_samples: Optional[int] = 1000,
        subrtn_policy: Optional[Algorithm] = None,
        subrtn_policy_snapshot_mode: Optional[str] = "latest",
        num_workers: Optional[int] = 1,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain
        :param policy: policy used for sampling the rollout, if subrtn_policy is not `None` this policy is not oly used
                       for generating the target domain rollouts, but also optimized in simulation
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param prior: distribution used by sbi as a prior
        :param posterior_nn_hparam: hyper parameters for creating the posterior"s density estimator
        :param sbi_subrtn_class: sbi algorithm calls for executing the LFI, e.g. SNPE
        :param summary_statistic: the method with which the observations for LFI are computed from the rollouts
                                  Possible options:
                                 `states` (uses all observed states from rollout),
                                 `final_state` (use the last observed state from the rollout), and
                                 `ramos` (summary statistics as proposed in  [1])
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_real_rollouts: number of real-world observation received by sbi, i.e. from every rollout exactly one
                                  observation is computed
        :param num_sim_per_real_rollout: number of simulations done by sbi per real-world observation received
        :param num_eval_samples: number of samples for evaluating the posterior in `eval_posterior()`
        :param subrtn_policy: algorithm which performs the optimization of the behavioral policy (and value-function)
        :param subrtn_policy_snapshot_mode: snapshot mode for saving during training of the subroutine
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call InterruptableAlgorithm's constructor
        super().__init__(num_checkpoints=2, save_dir=save_dir, max_iter=max_iter, policy=policy, logger=logger)

        self._env_sim = env_sim
        self._env_real = env_real
        self.dp_mapping = dp_mapping
        self.summary_statistic = summary_statistic.lower()
        self.posterior_nn_hparam = posterior_nn_hparam
        self.sbi_subrtn_class = sbi_subrtn_class
        self.sbi_training_hparam = dict()
        self.num_real_rollouts = num_real_rollouts
        self.num_sim_per_real_rollout = num_sim_per_real_rollout
        self.num_eval_samples = num_eval_samples
        self.num_workers = num_workers

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

        # Prepare simulator and prior for usage in sbi
        rollout_sampler = SimRolloutSamplerForSBI(self._env_sim, self._policy, self.dp_mapping, self.summary_statistic)
        self._sbi_simulator, self._sbi_prior = prepare_for_sbi(rollout_sampler, prior)

        # Create the algorithm instance used in sbi, e.g. SNPE-A/B/C or SNLE
        density_estimator = posterior_nn(**self.posterior_nn_hparam)  # can't be saved
        summary_writer = self.logger.printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self._sbi_subrtn = self.sbi_subrtn_class(
            prior=self._sbi_prior, density_estimator=density_estimator, summary_writer=summary_writer
        )

        # Save initial environments and the prior
        pyrado.save(self._env_sim, "env_sim", "pkl", self._save_dir)
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

    def step(self, snapshot_mode: str = None, meta_info: dict = None):
        # Save snapshot to save the correct iteration count
        self.save_snapshot()

        if self.curr_checkpoint == 0:
            observations_real = LFI.collect_real_observations(
                self.save_dir,
                self._env_real,
                self._policy,
                self.summary_statistic,
                prefix=f"iter_{self._curr_iter}",
                num_rollouts=self.num_real_rollouts,
            )
            if observations_real.ndim != 2 or observations_real.shape[0] != self.num_real_rollouts:
                raise pyrado.ShapeErr(
                    msg=f"The observations must be a 2-dim PyTorch tensor where the first dimension has as"
                    f"many entries as there are observations, but the shape is {observations_real.shape}!"
                )

            # First iteration
            if self._curr_iter == 0:
                # Sample parameters proposal, and simulate these parameters to obtain the observations
                domain_param, sim_output = simulate_for_sbi(
                    simulator=self._sbi_simulator,
                    proposal=self._sbi_prior,
                    num_simulations=self.num_sim_per_real_rollout,
                    simulation_batch_size=1,
                    num_workers=self.num_workers,
                )
                self._sbi_subrtn.append_simulations(
                    domain_param,
                    sim_output,
                    proposal=None,  # pass None if the parameters were sampled from the prior
                )
                self._cnt_samples += self.num_sim_per_real_rollout

                # Append the first set of observations
                pyrado.save(observations_real, "observations_real", "pt", self._save_dir)

            # Remaining training iterations
            else:
                posterior = pyrado.load(None, "posterior", "pt", self._save_dir, meta_info)

                for ro in observations_real:
                    posterior.set_default_x(ro)

                    # Sample parameters proposal, and simulate these parameters to obtain the observations
                    domain_param, sim_output = simulate_for_sbi(
                        simulator=self._sbi_simulator,
                        proposal=posterior,
                        num_simulations=self.num_sim_per_real_rollout,
                        simulation_batch_size=1,
                        num_workers=self.num_workers,  # leave it for now
                    )
                    self._sbi_subrtn.append_simulations(domain_param, sim_output, proposal=posterior)
                    self._cnt_samples += self.num_sim_per_real_rollout

                # Append and save all observations
                prev_observations = pyrado.load(None, "observations_real", "pt", self._save_dir)
                observations_real_hist = to.cat([prev_observations, observations_real], dim=0)
                pyrado.save(observations_real_hist, "observations_real", "pt", self._save_dir)

            self.reached_checkpoint()  # setting counter to 1

        if self.curr_checkpoint == 1:
            # Train the posterior
            self._sbi_subrtn.train(**self.sbi_training_hparam)
            posterior = (
                self._sbi_subrtn.build_posterior()
            )  # no need to pass density_estimator, since latest is used by default
            pyrado.save(posterior, "posterior", "pt", self._save_dir, meta_info=dict(prefix=f"iter_{self._curr_iter}"))
            pyrado.save(posterior, "posterior", "pt", self._save_dir, meta_info, use_state_dict=False)

            # Logging
            domain_param_eval, log_prob, _ = LFI.eval_posterior(
                posterior, observations_real, self.num_eval_samples, self._sbi_simulator, simulate_observations=False
            )
            self.logger.add_value("avg domain param", to.mean(domain_param_eval, dim=[0, 1]))
            self.logger.add_value("std domain param", to.std(domain_param_eval, dim=[0, 1]))
            self.logger.add_value("avg log prob", to.mean(log_prob))
            self.logger.add_value("num total samples", self._cnt_samples)  # here the samples are simulations

            self.reached_checkpoint()  # setting counter to 2

        if self.curr_checkpoint == 2:
            if self._subrtn_policy is not None:
                self.train_policy_sim(prefix=f"iter_{self._curr_iter}")

            self.reached_checkpoint()  # setting counter to 0

        # Save snapshot data
        self.make_snapshot(snapshot_mode, None, meta_info)

    @staticmethod
    def eval_posterior(
        posterior: DirectPosterior,
        observations_real: to.Tensor,
        num_samples: int,
        simulator: Callable = None,
        simulate_observations: bool = True,
    ) -> Tuple[to.Tensor, to.Tensor, Optional[to.Tensor]]:
        r"""
        Evaluates the posterior by computing parameter samples given observed data, its log probability
        and the simulated trajectory.

        :param posterior: posterior to evaluate, e.g. a normalizing flow, that samples domain parameters conditioned on
                          the provided observations
        :param observations_real: observations from the real-world rollouts a.k.a. $x_o$
        :param num_samples: number of samples to draw from the posterior
        :param simulator: simulator used during the sbi training procedure
        :param simulate_observations: create simulated observations using the domain parameters sampled from the
                                      posterior and same simulator as used during the sbi training procedure
        :return: domain parameters sampled form the posterior, log-probabilities of these domain parameters, and
                 optionally the simulated observations from the rollouts
        """

        def _eval_single_obs():
            """ Take the variables from the outer scope, and evaluate the posterior for one real-world observation. """
            # Compute the log probability
            log_prob[idx, :] = posterior.log_prob(domain_params[idx, :, :], x=observations_real[idx, :])

            # Simulate trajectories with the domain parameters from the posterior
            if simulate_observations:
                observations_sim[idx, :, :] = to.cat(
                    [simulator(domain_params[idx, s, :].unsqueeze(0)) for s in range(num_samples)], dim=0
                )

        if observations_real.ndim != 2:
            raise pyrado.ShapeErr(msg="The observations must be a 2-dim PyTorch tensor!")
        num_obs, dim_obs = observations_real.shape

        # Sample domain parameters from the normalizing flow
        domain_params = to.stack([posterior.sample((num_samples,), x=obs) for obs in observations_real], dim=0)
        if domain_params.shape[0] != num_obs or domain_params.shape[1] != num_samples:  # shape[2] = num_domain_param
            raise pyrado.ShapeErr(given=domain_params, expected_match=(num_obs, num_samples, -1))

        # Init containers
        log_prob = to.empty((num_obs, num_samples))
        observations_sim = to.empty((num_obs, num_samples, dim_obs)) if simulate_observations else None

        # Evaluate
        for idx in tqdm(
            range(num_obs),
            total=num_obs,
            desc="Evaluating posterior",
            unit="observations",
            file=sys.stdout,
            leave=False,
        ):
            _eval_single_obs()

        return domain_params, log_prob, observations_sim

    @staticmethod
    def collect_real_observations(
        save_dir: Optional[str],
        env: Union[RealEnv, SimEnv],
        policy: Policy,
        summary_statistic: str,
        prefix: str,
        num_rollouts: int,
    ) -> to.Tensor:
        """
        Roll-out a (behavioral) policy on the target system (real-world platform), and save the observations computed
        from the recorded rollouts.
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param summary_statistic: the method with which the observations for LFI are computed from the rollouts.
                                  Possible options:
                                 `states` (uses all observed states from rollout),
                                 `final_state` (use the last observed state from the rollout), and
                                 `ramos` (summary statistics as proposed in  [1])
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :return: 2-dim tensor of observations extracted from the rollouts, where the samples are along the first dim
        """
        if not (isinstance(inner_env(env), RealEnv) or isinstance(inner_env(env), SimEnv)):
            raise pyrado.TypeErr(given=inner_env(env), expected_type=[RealEnv, SimEnv])

        # Evaluate sequentially (necessary for sim-to-real experiments)
        rollout_worker = RealRolloutSamplerForSBI(env, policy, summary_statistic)
        observations_real = []
        for _ in tqdm(
            range(num_rollouts),
            total=num_rollouts,
            desc=Fore.CYAN + Style.BRIGHT + f"Executing {prefix}_policy" + Style.RESET_ALL,
            unit="rollouts",
            file=sys.stdout,
        ):
            observations_real.append(rollout_worker())

        # LFI.truncate_to_shortest(observations_real)

        # Stacked to tensor, samples along 1st dimension
        observations_real = to.stack(observations_real)

        # Optionally save the data
        if save_dir is not None:
            pyrado.save(observations_real, f"{prefix}_observations_real", "pt", save_dir)

        return observations_real

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            if self._subrtn_policy is None:
                # The policy is not being updated by a policy optimization subroutine
                pyrado.save(self._policy, "policy", "pt", self.save_dir, None)
            else:
                self._subrtn_policy.save_snapshot()

        else:
            raise pyrado.ValueErr(msg=f"{self.name} is not supposed be run as a subroutine!")

    def train_policy_sim(self, prefix: str) -> float:
        """
        Train a policy in simulation for given hyper-parameters from the domain randomizer.

        :param prefix: set a prefix to the saved file name by passing it to `meta_info`
        :return: estimated return of the trained policy in the target domain
        """
        # Set the domain randomizer
        # self._env_sim.adapt_randomizer(cand.detach().cpu().numpy())

        # Reset the subroutine algorithm which includes resetting the exploration
        # self._cnt_samples += self._subrtn_policy.sample_count
        self._subrtn_policy.reset()

        # Do a warm start if desired
        # self._subrtn_policy.init_modules(
        #     self.warmstart, policy_param_init=self.policy_param_init, valuefcn_param_init=self.valuefcn_param_init
        # )

        # Train a policy in simulation using the subroutine
        self._subrtn_policy.train(snapshot_mode=self._subrtn_policy_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        avg_ret_sim = self.eval_policy(
            None, self._env_sim, self._subrtn_policy.policy, prefix, self.num_eval_samples
        )
        return float(avg_ret_sim)

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
        rollout_sampler = SimRolloutSamplerForSBI(self._env_sim, self._policy, self.dp_mapping, self.summary_statistic)
        sbi_simulator, _ = prepare_for_sbi(rollout_sampler, state["_sbi_prior"])  # sbi_prior is fine as it is
        self.__dict__["_sbi_simulator"] = sbi_simulator

        # Reconstruct the tensorboard printer with the once from this algorithm
        summary_writer = state["_logger"].printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self.__dict__["_sbi_subrtn"]._summary_writer = summary_writer

        # Set the internal sbi construction callable to None
        self.__dict__["_sbi_subrtn"]._build_neural_net = None
