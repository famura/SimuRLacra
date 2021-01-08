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

import joblib
import os.path as osp
import torch as to
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe import PosteriorEstimator
from sbi.inference.base import simulate_for_sbi
from sbi.user_input.user_input_checks import prepare_for_sbi
from torch.distributions import Distribution
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Callable, Type, Union, Mapping

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.sbi_rollout_sampler import RolloutSamplerForSBI, RealRolloutSamplerForSBI
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.utils.checks import check_all_lengths_equal
from pyrado.utils.input_output import print_cbt


class LFI(Algorithm):
    """
    Learn a physically-grounded stochastic simulator using the sbi toolbox.

    TODO This class currently only works with posterior estimators and currently excludes likelihood- and density-ratio-estimators. This might be added later.
    """

    name: str = "lfi"  # TODO better acronym

    def __init__(
        self,
        save_dir: str,
        env_sim: SimEnv,
        env_real: Union[RealEnv, EnvWrapper],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        prior: Distribution,
        flow: Callable[[], DirectPosterior],
        sbi_subrtn_class: Type[PosteriorEstimator],
        max_iter: int,
        num_real_rollouts: int,
        num_sim_per_real_rollout: int,
        posterior: DirectPosterior = None,
        num_samples: int = 25,
        logger: Optional[StepLogger] = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain
        :param policy: behavioral policy  TODO fixed for now, eventually learned
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
        :param prior:
        :param flow:
        :param sbi_subrtn_class:
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param posterior:
        :param num_sim_per_real_rollout:
        :param num_real_rollouts:
        :param num_samples:
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        self._env_sim = env_sim
        self._env_real = env_real
        self.dp_mapping = dp_mapping
        self._posterior = posterior
        self._prior = prior
        self.num_real_rollouts = num_real_rollouts
        self.num_sim_per_real_rollout = num_sim_per_real_rollout
        self.num_samples = num_samples
        self.sbi_training_hparam = dict()

        # Prepare simulator and prior for usage in sbi
        self._rollout_sampler = RolloutSamplerForSBI(self._env_sim, self._policy, dp_mapping)
        self.sbi_simulator, self.batch_prior = prepare_for_sbi(self._rollout_sampler, self._prior)

        # Create the algorithm instance used in sbi, e.g. SNPE-A/B/C or SNLE
        summary_writer = self.logger.printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self.sbi_subrtn = sbi_subrtn_class(prior=self._prior, density_estimator=flow, summary_writer=summary_writer)

    @property
    def prior(self) -> Distribution:
        """ Get the prior. """
        return self._prior

    @property
    def posterior(self) -> Union[DirectPosterior, None]:
        """ Get the posterior. """
        return self._posterior

    @posterior.setter
    def posterior(self, posterior: DirectPosterior):
        """ Set the posterior. """
        if not isinstance(posterior, DirectPosterior):
            raise pyrado.TypeErr(given=posterior, expected_type=DirectPosterior)
        self._posterior = posterior
    
    @property
    def num_sbi_simulations(self) -> int:
        """ Get the number of simulations done per call of the sbi subroutine. """
        return self.num_sim_per_real_rollout * self.num_real_rollouts

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Trains the posterior using SNPE using observed rollouts and the prior distribution

        """
        # TODO: raise exception if flow, prior or inference is not given. In this case only the posterior is given
        #  and should only be used for evaluation

        observations_real = LFI.collect_real_observations(
            self.save_dir,
            self._env_real,
            self._policy,
            prefix=f"iter_{self._curr_iter}",
            num_rollouts=self.num_real_rollouts,
        )

        if observations_real.ndim != 2:
            raise pyrado.ShapeErr(msg="The observations must be a 2-dim PyTorch tensor!")

        # Logging
        # num_sim = 0
        # log_prob_hist = []
        # num_sim_hist = []

        # Set proposal prior
        proposal_prior = self.batch_prior

        # First iteration
        if self._curr_iter == 0:
            theta, x = simulate_for_sbi(
                simulator=self.sbi_simulator,
                proposal=proposal_prior,
                num_simulations=self.num_sbi_simulations,
                simulation_batch_size=1,
            )
            posterior_estimator = self.sbi_subrtn.append_simulations(theta, x)
            posterior_estimator.train(**self.sbi_training_hparam)
            self._posterior = self.sbi_subrtn.build_posterior()

        # Remaining training iterations
        else:
            for ro in observations_real:
                self._posterior.set_default_x(ro)
                theta, x = simulate_for_sbi(
                    simulator=self.sbi_simulator,
                    proposal=self._posterior,
                    num_simulations=self.num_sbi_simulations,
                    simulation_batch_size=1,
                )
                self.sbi_subrtn.append_simulations(theta, x)
                # num_sim += self.num_sbi_simulations

            _ = self.sbi_subrtn.train()
            self._posterior = self.sbi_subrtn.build_posterior()

        # Logging
        _, log_prob, _ = self.evaluate(
            rollouts_real=observations_real, num_samples=self.num_samples, compute_quantity={"log_prob": True}
        )
        # log_prob = log_prob.mean().squeeze()
        # log_prob_hist.append(log_prob)
        # num_sim_hist.append(num_sim)
        self.logger.add_value("avg log prob", to.mean(log_prob))
        self.logger.add_value("num simulations", self.num_sbi_simulations)

    def evaluate(
        self,
        rollouts_real: to.Tensor,
        num_samples: int = 1000,
        compute_quantity: dict = None,
    ):
        """
        Evaluates the posterior by calculating parameter samples given observed data, its log probability
        and the simulated trajectory.

        :param rollouts_real:
        :param num_samples:
        :param compute_quantity:
        :return:
        """
        compute_dict = {"log_prob": False, "sample_params": False, "sim_traj": False}
        if compute_quantity is not None:
            if not all(k in compute_dict for k in compute_quantity):
                raise pyrado.KeyErr(keys=list(compute_quantity.keys()))
            compute_dict.update(compute_quantity)

        if rollouts_real.dim() == 1:
            # add a dimension to treat single rollouts as a batch of data
            rollouts_real = rollouts_real.unsqueeze(0)
        if rollouts_real.dim() > 2:
            raise pyrado.ShapeErr(given=rollouts_real)

        # generate sample parameters
        prop_params = to.stack([self._posterior.sample((num_samples,), x=obs) for obs in rollouts_real], dim=0)
        # prop_params = [self._posterior.sample((num_samples,), x=obs) for obs in rollouts_real]

        log_prob, sim_traj = None, None
        num_obs = rollouts_real.shape[0]
        lenum_obs = rollouts_real.shape[1]
        if compute_dict["sim_traj"]:
            # sim_trajs = []
            sim_traj = to.empty((num_obs, num_samples, lenum_obs))
        if compute_dict["log_prob"]:
            # log_probs = []
            log_prob = to.empty((num_obs, num_samples))

        cnt = 0
        for o in range(num_obs):

            # compute log probability
            if compute_dict["log_prob"]:
                log_prob[o, :] = self._posterior.log_prob(prop_params[o, :, :], x=rollouts_real[o, :])
                # log_probs.append(self._posterior.log_prob(prop_params[o], x=rollouts_real[o]))

            # compute trajectories
            if compute_dict["sim_traj"]:
                for s in range(num_samples):
                    if not s % 10:
                        print(
                            "\r[trainum_lfi.py/evaluate_lfi] Observation: ({}|{}), Sample: ({}|{})".format(
                                o, num_obs, s, num_samples
                            ),
                            end="",
                        )
                    # compute trajectories for each observation and every sample
                    sim_traj[o, s, :] = self.sbi_simulator(prop_params[o, s, :].unsqueeze(0))
                    # sim_trajs.append(self.sbi_simulator(prop_params[o]))
            cnt += 1
        if not compute_dict["sample_params"]:
            prop_params = None
        return prop_params, log_prob, sim_traj

    @staticmethod
    def collect_real_observations(
        save_dir: Optional[str],
        env: [RealEnv, SimEnv],
        policy: Policy,
        prefix: str,
        num_rollouts: int,
        num_parallel_envs: int = 1,
    ) -> to.Tensor:
        """
        Roll-out a (behavioral) policy on the target system (real-world platform), and save the observations computed
        from the recorded rollouts.
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance
        :param policy: policy to evaluate
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_parallel_envs: number of environments for the parallel sampler (only used for SimEnv)
        :return: 2-dim tensor of observations extracted from the rollouts, where the samples are along the first dim
        """
        if not (isinstance(inner_env(env), RealEnv) or isinstance(inner_env(env), SimEnv)):
            raise pyrado.TypeErr(given=inner_env(env), expected_type=[RealEnv, SimEnv])

        if save_dir is not None:
            print_cbt(f"Executing {prefix}_policy ...", "c", bright=True)

        # Evaluate sequentially when conducting a sim-to-real experiment
        rollout_worker = RealRolloutSamplerForSBI(env, policy)
        obs_real = []
        for i in range(num_rollouts):
            obs_real.append(rollout_worker())

        # Optionally save the data
        if save_dir is not None:
            pyrado.save(obs_real, f"{prefix}_observations_real", "pkl", save_dir)

        # TODO our current comparison on traj level causes problems if the simulations are not equally long
        min_length = min([len(obs) for obs in obs_real])
        if not check_all_lengths_equal(obs_real):
            # Truncate the observations if necessary
            print_cbt("Needed to truncate.", "y", bright=True)
            for idx, obs in enumerate(obs_real):
                obs_real[idx] = obs[:min_length]

        # Return stacked tensor
        obs_real = to.stack(obs_real)
        return obs_real

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            joblib.dump(self._env_sim, osp.join(self.save_dir, "env_sim.pkl"))
            joblib.dump(self._env_real, osp.join(self.save_dir, "env_real.pkl"))
            pyrado.save(self._policy, "policy", "pt", self.save_dir, None)
            pyrado.save(self._posterior, "posterior", "pt", self._save_dir, meta_info, use_state_dict=False)
        else:
            raise pyrado.ValueErr(msg=f"{self.name} is not supposed be run as a subroutine!")
