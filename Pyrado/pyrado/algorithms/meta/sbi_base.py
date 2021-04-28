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

import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import numpy as np
import torch as to
from colorama import Fore, Style
from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe import PosteriorEstimator
from sbi.utils.user_input_checks import prepare_for_sbi
from tabulate import tabulate
from torch.distributions import Distribution, Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm, InterruptableAlgorithm
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper, DomainRandWrapperBuffer
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sbi_embeddings import Embedding
from pyrado.sampling.sbi_rollout_sampler import (
    RealRolloutSamplerForSBI,
    RecRolloutSamplerForSBI,
    SimRolloutSamplerForSBI,
)
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.data_types import merge_dicts
from pyrado.utils.input_output import completion_context, print_cbt


class SBIBase(InterruptableAlgorithm, ABC):
    """Base class for all Simulation-Based Inference (SBI) algorithms, wrapping the sbi package

    .. note::
        This class currently only works with (direct) posterior estimators and currently excludes likelihood- and
        density-ratio-estimators. This might be added later.

    .. seealso::
        [1] https://github.com/mackelab/sbi/blob/main/tutorials/03_multiround_inference.ipynb (multi-round sbi)
    """

    def __init__(
        self,
        num_checkpoints: int,
        init_checkpoint: int,
        save_dir: pyrado.PathLike,
        env_sim: SimEnv,
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

        :param num_checkpoints: total number of checkpoints
        :param init_checkpoint: initial value of the cyclic counter, defaults to 0, use negative values can to mark
                                sections that should only be executed once
        :param save_dir: directory to save the snapshots i.e. the results in
        :param env_sim: randomized simulation environment a.k.a. source domain
        :param env_real: real-world environment a.k.a. target domain, this can be a `RealEnv` (sim-to-real setting), a
                         `SimEnv` (sim-to-sim setting), or a directory to load a pre-recorded set of rollouts from
        :param policy: policy used for sampling the rollout, if subrtn_policy is not `None` this policy is not oly used
                       for generating the target domain rollouts, but also optimized in simulation
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param prior: distribution used by sbi as a prior
        :param subrtn_sbi_class: sbi algorithm calls for executing the LFI, e.g. SNPE
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
        :param use_rec_act: if `True` the recorded actions form the target domain are used to generate the rollout
                            during simulation (feed-forward). If `False` there policy is used to generate (potentially)
                            state-dependent actions (feed-back).
        :param num_sbi_rounds: set to an integer > 1 to use multi-round sbi. This way the posteriors (saved as
                               `..._round_NUMBER...` will be tailored to the data of that round, where `NUMBER`
                               counts up each round (modulo `num_real_rollouts`). If `num_sbi_rounds` = 1, the posterior
                               is called amortized (it has never seen any target domain data).
        :param num_eval_samples: number of samples for evaluating the posterior in `eval_posterior()`
        :param posterior_hparam: hyper parameters for creating the posterior's density estimator
        :param subrtn_sbi_training_hparam: `dict` forwarded to sbi's `PosteriorEstimator.train()` function like
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
        if not isinstance(inner_env(env_sim), SimEnv) or (
            isinstance(env_sim, DomainRandWrapper) and not isinstance(env_sim, ActDelayWrapper)
        ):
            raise pyrado.TypeErr(
                msg="The given env_sim must be a non-randomized simulation environment, "
                "except for wrappers that add a domain parameter!"
            )
        if isinstance(prior, Normal):
            raise pyrado.TypeErr(
                msg="The sbi framework requires MultivariateNormal instead of Normal distributions for the prior."
            )
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

        # Call InterruptableAlgorithm's constructor
        super().__init__(
            num_checkpoints=num_checkpoints,
            init_checkpoint=init_checkpoint,
            save_dir=save_dir,
            max_iter=max_iter,
            policy=policy,
            logger=logger,
        )

        self._env_sim_sbi = env_sim  # will be randomized explicitly by sbi
        self._env_sim_trn = DomainRandWrapperBuffer(deepcopy(env_sim), randomizer=None, selection="random")
        self._env_real = env_real
        self.dp_mapping = dp_mapping
        self._embedding = embedding
        self.subrtn_sbi_class = subrtn_sbi_class
        self.num_sim_per_round = num_sim_per_round
        self.num_real_rollouts = num_real_rollouts
        self.num_segments = num_segments
        self.len_segments = len_segments
        self.use_rec_act = use_rec_act
        self.num_sbi_rounds = num_sbi_rounds
        self.num_eval_samples = num_eval_samples or 10 * 2 ** len(dp_mapping)
        self.simulation_batch_size = simulation_batch_size
        self.normalize_posterior = normalize_posterior
        self.subrtn_sbi_training_hparam = subrtn_sbi_training_hparam or dict()
        self.subrtn_sbi_sampling_hparam = subrtn_sbi_sampling_hparam or dict()
        self.posterior_hparam = posterior_hparam or dict()
        self.thold_succ_subrtn = float(thold_succ_subrtn)
        self.max_subrtn_rep = 3  # number of tries to exceed thold_succ_subrtn during training in simulation
        self.num_workers = int(num_workers)

        # Temporary containers
        self._curr_data_real = None
        self._curr_domain_param_eval = None

        # Initialize sbi simulator and prior
        self._sbi_simulator = None  # to be set in step()
        self._sbi_prior = None  # to be set in step()
        self._setup_sbi(prior=prior)

        # Create the algorithm instance used in sbi, e.g. SNPE-A/B/C or SNLE
        density_estimator = utils.posterior_nn(**self.posterior_hparam)  # embedding for nflows is always nn.Identity
        summary_writer = self.logger.printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self._subrtn_sbi = self.subrtn_sbi_class(
            prior=self._sbi_prior, density_estimator=density_estimator, summary_writer=summary_writer
        )

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

        # Save initial environments, the embedding, and the prior
        pyrado.save(self._env_sim_trn, "env_sim.pkl", self._save_dir)
        pyrado.save(self._env_real, "env_real.pkl", self._save_dir)
        pyrado.save(embedding, "embedding.pt", self._save_dir)
        pyrado.save(prior, "prior.pt", self._save_dir)

    @property
    def subroutine_policy(self) -> Algorithm:
        """Get the policy optimization subroutine."""
        return self._subrtn_policy

    @property
    def subroutine_sbi(self) -> NeuralInference:
        """Get the system identification subroutine coming from the sbi module."""
        return self._subrtn_sbi

    @property
    def sbi_simulator(self) -> Optional[Callable]:
        """Get the simulator wrapped for sbi."""
        return self._sbi_simulator

    @property
    def embedding(self) -> Embedding:
        """Get the embedding used to compute the features from the rollouts."""
        return self._embedding

    def _setup_sbi(self, prior: Distribution, rollouts_real: Optional[List[StepSequence]] = None):
        """
        Prepare simulator and prior for usage in sbi.

        :param prior: distribution used by sbi as a prior
        :param rollouts_real: list of rollouts recorded from the target domain, which are used to sync the simulations'
                              initial states
        """
        rollout_sampler = SimRolloutSamplerForSBI(
            self._env_sim_sbi,
            self._policy,
            self.dp_mapping,
            self._embedding,
            self.num_segments,
            self.len_segments,
            rollouts_real,
            self.use_rec_act,
        )

        # Call sbi's preparation function
        self._sbi_simulator, self._sbi_prior = prepare_for_sbi(rollout_sampler, prior)

    def get_latest_proposal(self) -> Union[utils.BoxUniform, DirectPosterior]:
        """
        Get the latest proposal. This is either the prior, or the (amortized) posterior from the previous iteration.

        :return: latest proposal for simulating with sbi
        """
        if self._curr_iter == 0:
            proposal = self._sbi_prior
        else:
            proposal = pyrado.load("posterior.pt", self._save_dir, prefix=f"iter_{self._curr_iter - 1}")
        return proposal

    @abstractmethod
    def step(self, snapshot_mode: str, meta_info: dict = None):
        raise NotImplementedError

    @staticmethod
    @to.no_grad()
    def collect_data_real(
        save_dir: Optional[pyrado.PathLike],
        env: Union[Env, str],
        policy: Policy,
        embedding: Embedding,
        prefix: str,
        num_rollouts: int,
        num_segments: int = None,
        len_segments: int = None,
    ) -> Tuple[to.Tensor, List[StepSequence]]:
        """
        Roll-out a (behavioral) policy on the target system for later use with the sbi module, and save the data
        computed from the recorded rollouts.
        This method is static to facilitate evaluation of specific policies in hindsight.

        :param save_dir: directory to save the snapshots i.e. the results in, if `None` nothing is saved
        :param env: target environment for evaluation, in the sim-2-sim case this is another simulation instance,
                    in case you want to use pre-recorded rollouts pass the path to the parent folder as string
        :param policy: policy to evaluate
        :param embedding: embedding used for pre-processing the data before passing it to the posterior
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :param num_rollouts: number of rollouts to collect on the target system
        :param num_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param prefix: to control the saving for the evaluation of an initial policy, `None` to deactivate
        :return: data from the real-world rollouts a.k.a. set of $x_o$ of shape [num_iter, num_rollouts_per_iter,
                 time_series_length, dim_data], and the real-world rollouts
        """
        if not (isinstance(inner_env(env), RealEnv) or isinstance(inner_env(env), SimEnv) or isinstance(env, str)):
            raise pyrado.TypeErr(given=inner_env(env), expected_type=[RealEnv, SimEnv, str])

        # Evaluate sequentially (necessary for sim-to-real experiments)
        if isinstance(env, str):
            rollout_worker = RecRolloutSamplerForSBI(
                env, embedding, num_segments, len_segments, rand_init_rollout=False
            )
        else:
            rollout_worker = RealRolloutSamplerForSBI(env, policy, embedding, num_segments, len_segments)

        data_real = []
        rollouts_real = []
        for _ in tqdm(
            range(num_rollouts),
            total=num_rollouts,
            desc=Fore.CYAN + Style.BRIGHT + f"Collecting data using {prefix}_policy" + Style.RESET_ALL,
            unit="rollouts",
            file=sys.stdout,
        ):
            data, local_rollout = rollout_worker()
            data_real.append(data)
            rollouts_real.append(local_rollout)

        # Stacked to tensor of shape [1, num_rollouts * dim_feat]
        data_real = to.cat(data_real, dim=1)
        if data_real.shape != (1, num_rollouts * embedding.dim_output):
            raise pyrado.ShapeErr(given=data_real, expected_match=(1, num_rollouts * embedding.dim_output))

        # Optionally save the data
        if save_dir is not None:
            pyrado.save(data_real, "data_real.pt", save_dir, prefix=prefix)
            pyrado.save(rollouts_real, "rollouts_real.pkl", save_dir, prefix=prefix)

        return data_real, rollouts_real

    @staticmethod
    def load_posterior(
        load_dir: pyrado.PathLike,
        idx_iter: int = -1,
        idx_round: int = -1,
        obj: Optional[Any] = None,
        verbose: bool = False,
    ) -> Optional[DirectPosterior]:
        """
        Load the posterior of a given iteration (and round).

        :param load_dir: experiment's directory to crawl through
        :param idx_iter: iteration to load, to load the latest pass -1
        :param idx_round: round to load, to load the latest pass -1, ignored if the experiment was not multi-round
        :param obj: object for state dict loading, forwarded to `pyrado.load()`, by default no state dict loading
        :param verbose: if `True`, print the path of what has been loaded, forwarded to `pyrado.load()`
        :return: loaded sbi posterior, or `None` if there is no posterior with the given iteration / round index
        """
        if not os.path.isdir(load_dir):
            raise pyrado.PathErr(given=load_dir)
        if not isinstance(idx_iter, int):
            raise pyrado.TypeErr(given=idx_iter, expected_type=int)
        if not isinstance(idx_round, int):
            raise pyrado.TypeErr(given=idx_round, expected_type=int)

        if idx_iter == -1:
            # Check what is the latest iteration
            cnt_iter_max = -1
            for _, dirs, files in os.walk(load_dir):
                dirs.clear()  # prevents walk() from going into subdirectories
                for f in files:
                    if f.startswith("iter_") and f.endswith("_posterior.pt"):
                        cnt_iter = int(f[f.find("iter_") + len("iter_")])
                        cnt_iter_max = cnt_iter if cnt_iter > cnt_iter_max else cnt_iter_max
            idx_iter = cnt_iter_max

        # Check if the experiment was run in a multi-round setting
        multi_round_setting = False
        for _, dirs, files in os.walk(load_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            for f in files:
                if f.startswith("iter_") and "round" in f:
                    multi_round_setting = True
                    break

        if multi_round_setting:
            if idx_round == -1:
                # Check what is the latest round
                cnt_round_max = -1
                for _, dirs, files in os.walk(load_dir):
                    dirs.clear()  # prevents walk() from going into subdirectories
                    for f in files:
                        if f.startswith("iter_") and "round" in f and f.endswith("_posterior.pt"):
                            cnt_round = int(f[f.find("round_") + len("round_")])
                            cnt_round_max = cnt_round if cnt_round > cnt_round_max else cnt_round_max
                idx_round = cnt_round_max

        # Check before loading, and print a warning message if there can not be a posterior with the obtained indices
        if idx_iter == -1:
            print_cbt(f"Invalid iteration index {idx_iter}! Check if there is a posterior in {load_dir}.", "r")
        if idx_round == -1 and multi_round_setting:
            print_cbt(f"Invalid round index {idx_round}! Check if there is a posterior in {load_dir}.", "r")

        # Load the current posterior
        str_round = f"_round_{idx_round}" if multi_round_setting else ""
        try:
            posterior = pyrado.load(
                name=f"iter_{idx_iter}{str_round}_posterior.pt", load_dir=load_dir, obj=obj, verbose=verbose
            )
        except FileNotFoundError:
            print_cbt("No posterior was loaded.", "y")
            posterior = None

        return posterior

    @staticmethod
    @to.no_grad()
    def eval_posterior(
        posterior: DirectPosterior,
        data_real: to.Tensor,
        num_samples: int,
        calculate_log_probs: bool = True,
        normalize_posterior: bool = True,
        subrtn_sbi_sampling_hparam: Optional[dict] = None,
    ) -> Tuple[to.Tensor, Optional[to.Tensor]]:
        r"""
        Evaluates the posterior by computing parameter samples given observed data, its log probability
        and the simulated trajectory.

        :param posterior: posterior to evaluate, e.g. a normalizing flow, that samples domain parameters conditioned on
                          the provided data
        :param data_real: data from the real-world rollouts a.k.a. set of $x_o$ of shape
                          [num_iter, num_rollouts_per_iter, time_series_length, dim_data]
        :param num_samples: number of samples to draw from the posterior
        :param calculate_log_probs: if `True`, the log-probabilities are computed, else `None` is returned
        :param normalize_posterior: if `True`, the normalization of the posterior density is enforced by sbi
        :param subrtn_sbi_sampling_hparam: keyword arguments forwarded to sbi's `DirectPosterior.sample()` function
        :return: domain parameters sampled form the posterior of shape [batch_size, num_samples, dim_domain_param], as
                 well as the log-probabilities of these domain parameters
        """
        if not isinstance(data_real, to.Tensor) or data_real.ndim != 2:
            raise pyrado.ShapeErr(msg=f"The data must be a 2-dim PyTorch tensor, but is of shape {data_real.shape}!")

        batch_size, _ = data_real.shape

        # Sample domain parameters for all batches and stack them
        default_sampling_hparam = dict(
            mcmc_method="slice_np", mcmc_parameters=dict(warmup_steps=30)  # default: slice_np, 20
        )
        subrtn_sbi_sampling_hparam = merge_dicts([default_sampling_hparam, subrtn_sbi_sampling_hparam or dict()])
        domain_params = to.stack(
            [posterior.sample((num_samples,), x=x_o, **subrtn_sbi_sampling_hparam) for x_o in data_real],
            dim=0,
        )

        # Check shape
        if not domain_params.ndim == 3 or domain_params.shape[:2] != (batch_size, num_samples):
            raise pyrado.ShapeErr(
                msg=f"The sampled domain parameters must be a 3-dim tensor where the 1st dimension is {batch_size} and "
                f"the 2nd dimension is {num_samples}, but it is of shape {domain_params.shape}!"
            )

        # Compute the log probability if desired
        if calculate_log_probs:
            # Batch-wise computation and stacking
            with completion_context("Evaluating posterior", color="w"):
                log_probs = to.stack(
                    [
                        posterior.log_prob(dp, x=x_o, norm_posterior=normalize_posterior)
                        for dp, x_o in zip(domain_params, data_real)
                    ],
                    dim=0,
                )

            # Check shape
            if log_probs.shape != (batch_size, num_samples):
                raise pyrado.ShapeErr(given=log_probs, expected_match=(batch_size, num_samples))

        else:
            log_probs = None

        return domain_params, log_probs

    @staticmethod
    @to.no_grad()
    def get_ml_posterior_samples(
        dp_mapping: Mapping[int, str],
        posterior: DirectPosterior,
        data_real: to.Tensor,
        num_eval_samples: int,
        num_ml_samples: int = 1,
        calculate_log_probs: bool = True,
        normalize_posterior: bool = True,
        subrtn_sbi_sampling_hparam: Optional[dict] = None,
        return_as_tensor: bool = False,
    ) -> Tuple[Union[List[List[Dict]], to.Tensor], Optional[to.Tensor]]:
        r"""
        Evaluate the posterior conditioned on the data `data_real`, and extract the `num_ml_samples` most likely
        domain parameter sets.

        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param posterior: posterior to evaluate, e.g. a normalizing flow, that samples domain parameters conditioned on
                          the provided data
        :param data_real: data from the real-world rollouts a.k.a. set of $x_o$ of shape
                          [num_iter, num_rollouts_per_iter, time_series_length, dim_data]
        :param num_eval_samples: number of samples to draw from the posterior
        :param num_ml_samples: number of most likely samples, i.e. 1 equals argmax
        :param calculate_log_probs: if `True` the log-probabilities are computed, else `None` is returned
        :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
        :param subrtn_sbi_sampling_hparam: keyword arguments forwarded to sbi's `DirectPosterior.sample()` function
        :param return_as_tensor: if `True`, return the most likely domain parameter sets as a tensor of shape
                                 [num_iter, num_ml_samples, dim_domain_param], else as a list of `dict`
        :return: most likely domain parameters sets sampled form the posterior
        """
        if not isinstance(num_ml_samples, int) or num_ml_samples < 1:
            raise pyrado.ValueErr(given=num_ml_samples, g_constraint="0 (int)")

        # Evaluate the posterior
        domain_params, log_probs = SBIBase.eval_posterior(
            posterior,
            data_real,
            num_eval_samples,
            calculate_log_probs,
            normalize_posterior,
            subrtn_sbi_sampling_hparam,
        )

        # Extract the most likely domain parameter sets for every target domain data set
        domain_params_ml = []
        for idx_r in range(domain_params.shape[0]):
            idcs_ml = to.argsort(log_probs[idx_r, :], descending=True)
            idcs_sel = idcs_ml[:num_ml_samples]
            dp_vals = domain_params[idx_r, idcs_sel, :]

            if return_as_tensor:
                # Return as tensor
                domain_params_ml.append(dp_vals)

            else:
                # Return as dict
                dp_vals = np.atleast_1d(dp_vals.numpy())
                domain_param_ml = [dict(zip(dp_mapping.values(), dpv)) for dpv in dp_vals]
                domain_params_ml.append(domain_param_ml)

        if return_as_tensor:
            domain_params_ml = to.stack(domain_params_ml, dim=0)
            if not domain_params_ml.shape == (domain_params.shape[0], num_ml_samples, len(dp_mapping)):
                raise pyrado.ShapeErr(
                    given=domain_params_ml, expected_match=(domain_params.shape[0], num_ml_samples, len(dp_mapping))
                )

        else:
            # Check the first element
            if len(domain_params_ml[0]) != num_ml_samples or len(domain_params_ml[0][0]) != len(dp_mapping):
                raise pyrado.ShapeErr(
                    msg=f"The max likelihood domain parameter sets need to be of length {num_ml_samples}, but are "
                    f"{domain_params_ml[0]}, and the domain parameter sets need to be of length {len(dp_mapping)}, but "
                    f"are {len(domain_params_ml[0][0])}!"
                )

        return domain_params_ml, log_probs

    @staticmethod
    @to.no_grad()
    def eval_policy(
        save_dir: Optional[pyrado.PathLike],
        env: Env,
        policy: Policy,
        prefix: str,
        num_rollouts: int,
        num_workers: int = 1,
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
            for _ in range(num_rollouts):
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
            pyrado.save(rets_real, "returns_real.pt", save_dir, prefix=prefix)
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
        :param prefix: set a prefix to the saved file name, use "" for no prefix
        :return: estimated return of the trained policy in the target domain
        """
        if not (domain_params.ndim == 2 and domain_params.shape[1] == len(self.dp_mapping)):
            raise pyrado.ShapeErr(given=domain_params, expected_match=(-1, len(self.dp_mapping)))

        # Insert the domain parameters into the wrapped environment's buffer
        self.fill_domain_param_buffer(self._env_sim_trn, self.dp_mapping, domain_params)

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

        # Do a warm start if desired, but randomly reset the policy parameters if training failed once
        # self._subrtn_policy.init_modules(
        #     self.warmstart and cnt_rep == 0,
        #     policy_param_init=self.policy_param_init,
        #     valuefcn_param_init=self.valuefcn_param_init,
        # )

        # Train a policy in simulation using the subroutine
        self._subrtn_policy.train(snapshot_mode=self._subrtn_policy_snapshot_mode, meta_info=dict(prefix=prefix))

        # Return the estimated return of the trained policy in simulation
        assert len(self._env_sim_trn.buffer) == self.num_eval_samples
        self._env_sim_trn.ring_idx = 0  # don't reset the buffer to eval on the same domains as trained
        avg_ret_sim = self.eval_policy(
            None, self._env_sim_trn, self._subrtn_policy.policy, prefix, self.num_eval_samples
        )
        return float(avg_ret_sim)

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            if self._subrtn_policy is None:
                # The policy is not being updated by a policy optimization subroutine
                pyrado.save(self._policy, "policy.pt", self.save_dir, use_state_dict=True)
            else:
                self._subrtn_policy.save_snapshot()

        else:
            raise pyrado.ValueErr(msg=f"{self.name} is not supposed be run as a subroutine!")

    def __getstate__(self):
        # Remove the unpickleable sbi-related members from this algorithm instance
        tmp_sbi_simulator = self.__dict__.pop("_sbi_simulator")
        tmp_subrtn_sbi_summary_writer = self.__dict__["_subrtn_sbi"].__dict__.pop("_summary_writer")
        tmp_subrtn_sbi_build_neural_net = self.__dict__["_subrtn_sbi"].__dict__.pop("_build_neural_net")

        # Remove the policy optimization subroutine, since it contains non-leaf tensors. These cause an error during the
        # subsequent deepcopying
        tmp_subrtn_policy = self.__dict__.pop("_subrtn_policy", None)

        # Call Algorithm's __getstate__() without the unpickleable sbi-related members
        state_dict = super().__getstate__()

        # Make a deep copy of the state dict such that we can return the pickleable version and insert the sbi variables
        state_dict_copy = deepcopy(state_dict)

        # Inset them back
        self.__dict__["_sbi_simulator"] = tmp_sbi_simulator
        self.__dict__["_subrtn_sbi"]._summary_writer = tmp_subrtn_sbi_summary_writer
        self.__dict__["_subrtn_sbi"]._build_neural_net = tmp_subrtn_sbi_build_neural_net
        self.__dict__["_subrtn_policy"] = tmp_subrtn_policy

        return state_dict_copy

    def __setstate__(self, state):
        # Call Algorithm's __setstate__()
        super().__setstate__(state)

        # Reconstruct the simulator for sbi
        try:
            rollouts_real = pyrado.load("rollouts_real.pkl", self._save_dir, prefix=f"iter_{self._curr_iter}")
        except FileNotFoundError:
            try:
                rollouts_real = pyrado.load("rollouts_real.pkl", self._save_dir, prefix=f"iter_{self._curr_iter - 1}")
            except (FileNotFoundError, RuntimeError, pyrado.PathErr, pyrado.TypeErr, pyrado.ValueErr):
                rollouts_real = None
        self._setup_sbi(state["_sbi_prior"], rollouts_real)  # sbi_prior is fine as it is

        # Reconstruct the tensorboard printer with the once from this algorithm
        summary_writer = state["_logger"].printers[2].writer
        assert isinstance(summary_writer, SummaryWriter)
        self.__dict__["_subrtn_sbi"]._summary_writer = summary_writer

        # Set the internal sbi construction callable to None
        self.__dict__["_subrtn_sbi"]._build_neural_net = None
