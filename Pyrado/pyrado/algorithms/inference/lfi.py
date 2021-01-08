from typing import Optional, Callable, Type, Sequence, Tuple, Iterable, Union

import joblib
import pyrado
import torch as to
import os.path as osp

from pyrado.algorithms.episodic.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.algorithms.inference.rolloutsamplerforsbibase import RolloutSamplerForSBIBase
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger, TensorBoardPrinter, LoggerAware
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe import PosteriorEstimator
from sbi.inference.base import simulate_for_sbi
from sbi.user_input.user_input_checks import prepare_for_sbi
from torch.distributions import Distribution


class EnvSimulator(Callable):
    """
    Mapping from the environment system parameters to a trajectory-based rollout using a control-policy.
    """

    def __init__(self, env: Env, policy: Policy, param_names: list, strategy="states"):
        self.name = env.name
        self.env = env
        self.policy = policy
        self.param_names = param_names
        self.strategy = strategy
        self.transformed_representation = False

    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            reset_kwargs=dict(domain_param=dict(zip(self.param_names, params.squeeze().numpy()))),
        )
        if self.transformed_representation:
            ro = self.transform_data(ro, strategy=self.strategy)
        # return to.tensor(ro.observations).view(-1, 1).squeeze().to(dtype=to.float32)
        return ro

    def transform_data(self, ro: StepSequence, strategy: str = "summary"):
        if strategy is "states":
            context_strat = self.states_representation
        elif strategy is "summary":
            # calculate summary statistics
            context_strat = self.summary_statistics
        else:
            raise pyrado.ValueErr(given=strategy)
        return context_strat(ro)

    @staticmethod
    def summary_statistics(ro: StepSequence) -> to.Tensor:
        """
        Calculating summary statistics based on BayesSim
        """

        # calculate
        return 0

    @staticmethod
    def states_representation(ro: StepSequence) -> to.Tensor:
        return to.tensor(ro.observations).view(-1, 1).squeeze().to(dtype=to.float32)


class LFI(LoggerAware):
    """
    SBI-Wrapper.
    This class currently only works with posterior estimators and currently excludes
    likelihood- and density-ratio-estimators. This might be added later.
    Examplary file in '/pyrado/scripts/lfi/....py'
    """

    def __init__(
        self,
        save_dir: str,
        simulator: RolloutSamplerForSBIBase,
        prior: Distribution,
        inference: Type[PosteriorEstimator] = None,
        flow: Callable[[], DirectPosterior] = None,
        posterior: DirectPosterior = None,
        max_iter: int = 5,
        num_sim: int = 5,
        num_samples=25,
        logger: Optional[StepLogger] = None,
        save_name: str = "algo",
    ):
        self._save_dir = save_dir
        self.posterior = posterior
        self.simulator = simulator
        self.simulator.set_representation(True)
        self.prior = prior
        self._max_iter = max_iter
        self._curr_iter = 0
        self._num_sim = num_sim
        self._num_samples = num_samples
        self._cnt_samples = 0
        self.batch_simulator, self.batch_prior = prepare_for_sbi(self.simulator, self.prior)

        self._save_name = save_name

        if logger is not None:
            self._logger = logger

        # sbi should use the same summary writer as this algo
        summary_writer = None
        for p in self.logger.printers:
            if isinstance(p, TensorBoardPrinter):
                summary_writer = p.writer

        self.inference = None
        if inference is not None:
            self.inference = inference(prior=self.prior, density_estimator=flow, summary_writer=summary_writer)

    def set_posterior(self, posterior: DirectPosterior):
        """
        Set posterior from the outside if a learned model exists.
        """
        self.posterior = posterior

    def step(self, snapshot_mode: str, meta_info: dict = None, logging=False, data_representation: str = None):
        """
        Trains the posterior using SNPE using observed rollouts and the prior distribution
        """
        # TODO: raise exception if flow, prior or inference is not given. In this case only the posterior is given
        #  and should only be used for evaluation

        # get real-world rollouts from meta_info
        if "rollouts_real" not in meta_info.keys():
            raise pyrado.KeyErr(keys="rollouts_real")
        rollouts_real = meta_info["rollouts_real"]

        if not isinstance(rollouts_real, Sequence):
            raise pyrado.TypeErr(given=rollouts_real)
        if not isinstance(rollouts_real[0], StepSequence):
            raise pyrado.ShapeErr(given=rollouts_real[0])

        # obs_context = self.transform_data(rollouts=rollouts_real, strategy=data_representation)
        obs_context = []
        for ro in rollouts_real:
            obs_context.append(self.simulator.transform_data(ro))
        obs_context = to.stack(obs_context)

        # logging
        n_sim = 0
        log_probs = []
        n_simulations = []

        # set proposal prior
        proposal_prior = self.batch_prior

        # first iteration
        if self._curr_iter == 0:
            theta, x = simulate_for_sbi(
                self.batch_simulator,
                proposal_prior,
                num_simulations=self._num_sim * obs_context.shape[0],
                simulation_batch_size=1,
            )

            _ = self.inference.append_simulations(theta, x).train()
            self.posterior = self.inference.build_posterior()

            if logging:
                _, log_prob, _ = self.evaluate(
                    meta_info=dict(rollouts_real=rollouts_real),
                    num_samples=self._num_samples,
                    compute_quantity={"log_prob": True},
                )
                log_prob = log_prob.mean().squeeze()
                n_sim += self._num_sim
                n_simulations.append(n_sim)
                log_probs.append(log_prob)
                self.logger.add_value("Mean Log Probability", log_prob)
                self.logger.add_value("Number of Simulations", to.tensor(n_sim))

            self._curr_iter += 1

            self.logger.add_value("Current Iteration", self._curr_iter)

        # remaining training steps
        while self._curr_iter < self._max_iter:
            for ro in obs_context:
                self.posterior.set_default_x(ro)
                theta, x = simulate_for_sbi(
                    self.batch_simulator, self.posterior, num_simulations=self._num_sim, simulation_batch_size=1
                )
                self.inference.append_simulations(theta, x)
                n_sim += self._num_sim

            _ = self.inference.train()
            # set proposal prior
            self.posterior = self.inference.build_posterior()

            if logging:
                _, log_prob, _ = self.evaluate(
                    meta_info=dict(rollouts_real=rollouts_real),
                    num_samples=self._num_samples,
                    compute_quantity={"log_prob": True},
                )
                log_prob = log_prob.mean().squeeze()
                log_probs.append(log_prob)
                n_simulations.append(n_sim)
                self.logger.add_value("Mean Log Probability", log_prob)
                self.logger.add_value("Number of Simulations", to.tensor(n_sim))

                meta_info = {
                    **meta_info,
                    **dict(zip(["log_probs", "n_simulations"], [to.stack(log_probs), to.tensor(n_simulations)])),
                }

            self._curr_iter += 1
            self.logger.add_value("Current Iteration", self._curr_iter)
            self.make_snapshot(snapshot_mode=snapshot_mode, meta_info=meta_info)

    def evaluate(
        self, meta_info: dict, num_samples: int = 1000, compute_quantity: dict = None, data_representation: str = None
    ):
        """
        Evaluates the posterior by calculating parameter samples given observed data, its log probability
        and the simulated trajectory.
        """

        compute_dict = {"log_prob": False, "sample_params": False, "sim_traj": False}
        if compute_quantity is not None:
            if not all(k in compute_dict for k in compute_quantity):
                raise pyrado.KeyErr(keys=list(compute_quantity.keys()))
            compute_dict.update(compute_quantity)

        # get real-world rollouts from meta_info
        if "rollouts_real" not in meta_info.keys():
            raise pyrado.KeyErr(keys="rollouts_real")
        rollouts_real = meta_info["rollouts_real"]

        if not isinstance(rollouts_real, Sequence):
            raise pyrado.TypeErr(given=rollouts_real)
        if not isinstance(rollouts_real[0], StepSequence):
            raise pyrado.ShapeErr(given=rollouts_real[0])

        obs_context = []
        for ro in rollouts_real:
            obs_context.append(self.simulator.transform_data(ro))
        obs_context = to.stack(obs_context)

        # generate sample parameters
        prop_params = to.stack([self.posterior.sample((num_samples,), x=obs) for obs in obs_context], dim=0)
        # prop_params = [self.posterior.sample((num_samples,), x=obs) for obs in rollouts_real]

        log_prob, sim_traj = None, None
        num_obs = obs_context.shape[0]
        len_obs = obs_context.shape[1]
        if compute_dict["sim_traj"]:
            # sim_trajs = []
            sim_traj = to.empty((num_obs, num_samples, len_obs))
        if compute_dict["log_prob"]:
            # log_probs = []
            log_prob = to.empty((num_obs, num_samples))

        cnt = 0
        for o in range(num_obs):

            # compute log probability
            if compute_dict["log_prob"]:
                log_prob[o, :] = self.posterior.log_prob(prop_params[o, :, :], x=obs_context[o, :])
                # log_probs.append(self.posterior.log_prob(prop_params[o], x=rollouts_real[o]))

            # compute trajectories
            if compute_dict["sim_traj"]:
                for s in range(num_samples):
                    if not s % 10:
                        print(
                            "\r[train_lfi.py/evaluate_lfi] Observation: ({}|{}), Sample: ({}|{})".format(
                                o, num_obs, s, num_samples
                            ),
                            end="",
                        )
                    # compute trajectories for each observation and every sample
                    sim_traj[o, s, :] = self.batch_simulator(prop_params[o, s, :].unsqueeze(0))
                    # sim_trajs.append(self.batch_simulator(prop_params[o]))
            cnt += 1
        if not compute_dict["sample_params"]:
            prop_params = None
        return prop_params, log_prob, sim_traj

    def make_snapshot(self, snapshot_mode: str, meta_info: dict = None):
        """
        Make a snapshot of the training progress.
        This method is called from the subclasses and delegates to the custom method `save_snapshot()`.

        :param snapshot_mode: determines when the snapshots are stored (e.g. on every iteration or on new highscore)
        :param meta_info: is not `None` if this algorithm is run as a subroutine of a meta-algorithm,
                          contains a dict of information about the current iteration of the meta-algorithm
        """
        self.save_snapshot(meta_info)
        if snapshot_mode == "latest":
            self.save_snapshot(meta_info)
        else:
            raise pyrado.ValueErr(given=snapshot_mode, eq_constraint="'latest', 'best', or 'no'")

    def save_snapshot(self, meta_info: dict = None):
        # joblib.dump(self, osp.join(self._save_dir, f"{self._save_name}.pkl"))

        pyrado.save(self.posterior, "posterior", "pt", self._save_dir, meta_info, use_state_dict=False)

        if "rollouts_real" not in meta_info:
            raise pyrado.KeyErr(keys="rollouts_real", container=meta_info)
        pyrado.save(meta_info["rollouts_real"], "rollouts_real", "pkl", self._save_dir, meta_info)

        if "log_probs" in meta_info:
            pyrado.save(meta_info["log_probs"], "log_probs", "pkl", self._save_dir)
            pyrado.save(meta_info["n_simulations"], "n_simulations", "pkl", self._save_dir)
