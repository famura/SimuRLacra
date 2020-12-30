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
import numpy as np
import os.path as osp
from collections.abc import Iterable
from functools import partial
from itertools import product
from scipy.ndimage import gaussian_filter1d
from typing import Callable, Sequence, Tuple, Union, Optional

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.policies.base import Policy
from pyrado.policies.special.domain_distribution import DomainDistrParamPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult, ParameterSample
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.utils import gen_ordered_batch_idcs
from pyrado.utils.checks import check_all_equal
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import UnitCubeProjector


class SysIdViaEpisodicRL(Algorithm):
    """
    Wrapper to frame black-box system identification as an episodic reinforcement learning problem

    .. note::
        This algorithm was designed as a subroutine of SimOpt. However, it could also be used independently.
    """

    name: str = "sysiderl"
    iteration_key: str = "sysiderl_iteration"  # logger's iteration key

    def __init__(
        self,
        subrtn: ParameterExploring,
        behavior_policy: Policy,
        num_rollouts_per_distr: int,
        metric: Union[Callable[[np.ndarray], np.ndarray], None],
        obs_dim_weight: Union[list, np.ndarray],
        std_obs_filt: Optional[int] = 5,
        w_abs: float = 0.5,
        w_sq: float = 1.0,
        num_workers: int = 4,
        base_seed: int = 1001,
    ):
        """
        Constructor

        :param subrtn: wrapped algorithm to fit the domain parameter distribution
        :param behavior_policy: lower level policy used to generate the rollouts
        :param num_rollouts_per_distr: number of rollouts per domain distribution parameter set
        :param metric: functional mapping from differences in observations to value
        :param obs_dim_weight: (diagonal) weight matrix for the different observation dimensions for the default metric
        :param std_obs_filt: number of standard deviations for the Gaussian filter applied to the observaitons
        :param w_abs: weight for the mean absolute errors for the default metric
        :param w_sq: weight for the mean squared errors for the default metric
        :param num_workers: number of environments for parallel sampling
        :param base_seed: seed to set for the parallel sampler in every iteration
        """
        if not isinstance(subrtn, ParameterExploring):
            raise pyrado.TypeErr(given=subrtn, expected_type=ParameterExploring)
        if not isinstance(subrtn.env, MetaDomainRandWrapper):
            raise pyrado.TypeErr(given=subrtn.env, expected_type=MetaDomainRandWrapper)
        if not isinstance(subrtn.policy, DomainDistrParamPolicy):
            raise pyrado.TypeErr(given=subrtn.policy, expected_type=DomainDistrParamPolicy)
        if not isinstance(behavior_policy, Policy):
            raise pyrado.TypeErr(given=behavior_policy, expected_type=Policy)
        if subrtn.policy.num_param != len(subrtn.env.mapping):
            raise pyrado.ShapeErr(
                msg=f"Number of policy parameters {subrtn.policy.num_param} does not match the"
                f"number of domain distribution parameters {len(subrtn.env.mapping)}!"
            )
        if subrtn.sampler.num_rollouts_per_param != 1:
            # Only sample one rollout in every domain. This is possible since we are synchronizing the init state.
            raise pyrado.ValueErr(given=subrtn.sampler.num_rollouts_per_param, eq_constraint="1")
        if num_rollouts_per_distr < 2:
            raise pyrado.ValueErr(given=num_rollouts_per_distr, g_constraint="1")
        if len(obs_dim_weight) != subrtn.env.obs_space.flat_dim:
            raise pyrado.ShapeErr(given=obs_dim_weight, expected_match=subrtn.env.obs_space)

        # Call Algorithm's constructor
        super().__init__(subrtn.save_dir, subrtn.max_iter, subrtn.policy, subrtn.logger)

        self._subrtn = subrtn
        self._subrtn.save_name = "subrtn"
        self._behavior_policy = behavior_policy
        self.obs_dim_weight = np.diag(obs_dim_weight)  # weighting factor between the different observations
        self.std_obs_filt = std_obs_filt
        if metric is None or metric == "None":
            self.metric = partial(
                self.weighted_l1_l2_metric, w_abs=w_abs, w_sq=w_sq, obs_dim_weight=self.obs_dim_weight
            )
        else:
            self.metric = metric

        # Get and optionally clip the observation bounds of the environment
        elb, eub = subrtn.env.obs_space.bound_lo, subrtn.env.obs_space.bound_up
        elb, eub = self.override_obs_bounds(elb, eub, subrtn.env.obs_space.labels)
        self.obs_normalizer = UnitCubeProjector(bound_lo=elb, bound_up=eub)

        # Create the sampler used to execute the same policy as on the real system in the meta-randomized env
        self.base_seed = base_seed
        self.behavior_sampler = ParallelRolloutSampler(
            self._subrtn.env, self._behavior_policy, num_workers=num_workers, min_rollouts=1, seed=base_seed
        )
        self.num_rollouts_per_distr = num_rollouts_per_distr

    @property
    def subrtn(self) -> ParameterExploring:
        """ Get the subroutine used for updating the domain parameter distribution. """
        return self._subrtn

    def reset(self, seed: int = None):
        # Reset internal variables inherited from Algorithm
        self._curr_iter = 0
        self._cnt_samples = 0
        self._highest_avg_ret = -pyrado.inf

        # Forward to subroutine
        self._subrtn.reset(seed)

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if "rollouts_real" not in meta_info:
            raise pyrado.KeyErr(keys="rollouts_real", container=meta_info)
        if "init_state" not in meta_info["rollouts_real"][0].rollout_info:  # checking the first element is sufficient
            raise pyrado.KeyErr(keys="init_state", container=meta_info["rollouts_real"][0].rollout_info)

        # Extract the initial states from the real rollouts
        rollouts_real = meta_info["rollouts_real"]
        init_states_real = [ro.rollout_info["init_state"] for ro in rollouts_real]

        # Sample new policy parameters a.k.a domain distribution parameters
        param_sets = self._subrtn.expl_strat.sample_param_sets(
            nominal_params=self._subrtn.policy.param_values,
            num_samples=self._subrtn.pop_size,
            include_nominal_params=True,
        )

        # Iterate over every domain parameter distribution. We basically mimic the ParameterExplorationSampler here,
        # but we need to adapt the randomizer (and not just the domain parameters) por every policy param set
        param_samples = []
        loss_hist = []
        for idx_ps, ps in enumerate(param_sets):
            # Update the randomizer to use the new
            new_ddp_vals = self._subrtn.policy.transform_to_ddp_space(ps)
            self._subrtn.env.adapt_randomizer(domain_distr_param_values=new_ddp_vals.detach().cpu().numpy())
            self._subrtn.env.randomizer.randomize(num_samples=self.num_rollouts_per_distr)
            sampled_domain_params = self._subrtn.env.randomizer.get_params()

            # Sample the rollouts
            self.behavior_sampler.set_seed(self.base_seed)
            rollouts_sim = self.behavior_sampler.sample(init_states_real, sampled_domain_params, eval=True)

            # Iterate over simulated rollout with the same initial state
            for idx_real, idcs_sim in enumerate(
                gen_ordered_batch_idcs(self.num_rollouts_per_distr, len(rollouts_sim), sorted=True)
            ):
                # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
                ros_real_tr, ros_sim_tr = self.truncate_rollouts(
                    [rollouts_real[idx_real]], rollouts_sim[slice(idcs_sim[0], idcs_sim[-1] + 1)]
                )

                # Check the validity of the initial states. The domain parameters will be different.
                assert len(ros_real_tr) == len(ros_sim_tr) == len(idcs_sim)
                assert check_all_equal([r.rollout_info["init_state"] for r in ros_real_tr])
                assert check_all_equal([r.rollout_info["init_state"] for r in ros_sim_tr])
                assert all(
                    [
                        np.allclose(r.rollout_info["init_state"], s.rollout_info["init_state"])
                        for r, s in zip(ros_real_tr, ros_sim_tr)
                    ]
                )

                # Compute the losses
                losses = np.asarray([self.loss_fcn(ro_r, ro_s) for ro_r, ro_s in zip(ros_real_tr, ros_sim_tr)])

                if np.all(losses == 0.0):
                    raise pyrado.ValueErr(
                        msg="All SysIdViaEpisodicRL losses are equal to zero! Most likely the domain"
                        "randomization is too extreme, such that every trajectory is done after"
                        "one step. Check the exploration strategy."
                    )

                # Handle zero losses by setting them to the maximum current loss
                losses[losses == 0] = np.max(losses)
                loss_hist.extend(losses)

                # We need to assign the loss value to the simulated rollout, but this one can be of a different
                # length than the real-world rollouts as well as of different length than the original
                # (non-truncated) simulated rollout. Thus, we simply write the loss value into the first step.
                for i, l in zip(range(idcs_sim[0], idcs_sim[-1] + 1), losses):
                    rollouts_sim[i].rewards[:] = 0.0
                    rollouts_sim[i].rewards[0] = -l

            # Collect the results
            param_samples.append(ParameterSample(params=ps, rollouts=rollouts_sim))

        # Bind the parameter samples and their rollouts in the usual container
        param_samp_res = ParameterSamplingResult(param_samples)
        self._cnt_samples += sum([len(ro) for pss in param_samp_res for ro in pss.rollouts])

        # Log metrics computed from the old policy (before the update)
        loss_hist = np.asarray(loss_hist)
        self.logger.add_value("min sysid loss", np.min(loss_hist), 6)
        self.logger.add_value("median sysid loss", np.median(loss_hist), 6)
        self.logger.add_value("avg sysid loss", np.mean(loss_hist), 6)
        self.logger.add_value("max sysid loss", np.max(loss_hist), 6)
        self.logger.add_value("std sysid loss", np.std(loss_hist), 6)

        # Extract the best policy parameter sample for saving it later
        self._subrtn.best_policy_param = param_samp_res.parameters[np.argmax(param_samp_res.mean_returns)].clone()

        # Update the wrapped algorithm's update method
        self._subrtn.update(param_samp_res, ret_avg_curr=param_samp_res[0].mean_undiscounted_return)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(np.max(param_samp_res.mean_returns)), meta_info)

    @staticmethod
    def override_obs_bounds(bound_lo: np.ndarray, bound_up: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Default overriding method for the bounds of an observation space. This is necessary when the observations
        are scaled with their range, e.g. to compare a deviation over different kinds of abservations like position and
        annular velocity. Thus, infinite bounds are not feasible.

        :param bound_lo: lower bound of the observation space
        :param bound_up: upper bound of the observation space
        :return: clipped lower and upper bound
        """
        bound_lo = ObsNormWrapper.override_bounds(bound_lo, {"theta_dot": -20.0, "alpha_dot": -20.0}, labels)
        bound_up = ObsNormWrapper.override_bounds(bound_up, {"theta_dot": 20.0, "alpha_dot": 20.0}, labels)
        return bound_lo, bound_up

    @staticmethod
    def weighted_l1_l2_metric(err: np.ndarray, w_abs: float, w_sq: float, obs_dim_weight: np.ndarray):
        """
        Compute the weighted linear combination of the observation error's MAE and MSE, averaged over time

        .. note::
            In contrast to [1], we are using the mean absolute error and the mean squared error instead of the L1 and
            the L2 norm. The reason for this is that longer time series would be punished otherwise.

        :param err: error signal with time steps along the first dimension
        :param w_abs: weight for the mean absolute errors
        :param w_sq: weight for the mean squared errors
        :param obs_dim_weight: (diagonal) weight matrix for the different observation dimensions
        :return: weighted linear combination of the error's MAE and MSE, averaged over time
        """
        err_w = np.matmul(err, obs_dim_weight)
        return w_abs * np.mean(np.abs(err_w), axis=0) + w_sq * np.mean(np.power(err_w, 2), axis=0)

    def loss_fcn(self, rollout_real: StepSequence, rollout_sim: StepSequence) -> float:
        """
        Compute the discrepancy between two time sequences of observations given metric.
        Be sure to align and truncate the rollouts beforehand.

        :param rollout_real: (concatenated) real-world rollout containing the observations
        :param rollout_sim: (concatenated) simulated rollout containing the observations
        :return: discrepancy cost summed over the observation dimensions
        """
        if len(rollout_real) != len(rollout_sim):
            raise pyrado.ShapeErr(given=rollout_real, expected_match=rollout_sim)

        # Extract the observations
        real_obs = rollout_real.get_data_values("observations", truncate_last=True)
        sim_obs = rollout_sim.get_data_values("observations", truncate_last=True)

        # Filter the observations
        real_obs = gaussian_filter1d(real_obs, self.std_obs_filt, axis=0)
        sim_obs = gaussian_filter1d(sim_obs, self.std_obs_filt, axis=0)

        # Normalize the signals
        real_obs_norm = self.obs_normalizer.project_to(real_obs)
        sim_obs_norm = self.obs_normalizer.project_to(sim_obs)

        # Compute loss based on the error
        loss_per_obs_dim = self.metric(real_obs_norm - sim_obs_norm)
        assert len(loss_per_obs_dim) == real_obs.shape[1]
        assert all(loss_per_obs_dim >= 0)
        return sum(loss_per_obs_dim)

    @staticmethod
    def truncate_rollouts(
        rollouts_real: Sequence[StepSequence], rollouts_sim: Sequence[StepSequence], replicate: bool = True
    ) -> Tuple[Sequence[StepSequence], Sequence[StepSequence]]:
        """
        In case (some of the) rollouts failed or succeed in one domain, but not in the other, we truncate the longer
        observation sequence. When truncating, we compare every of the M real rollouts to every of the N simulated
        rollouts, thus replicate the real rollout N times and the simulated rollouts M times.

        :param rollouts_real: M real-world rollouts of different length if `replicate = True`, else K real-world
                              rollouts of different length
        :param rollouts_sim: N simulated rollouts of different length if `replicate = True`, else K simulated
                              rollouts of different length
        :param replicate: if `False` the i-th rollout from `rollouts_real` is (only) compared with the i-th rollout from
                          `rollouts_sim`, in this case the number of rollouts and the initial states have to match
        :return: MxN real-world rollouts and MxN simulated rollouts of equal length if `replicate = True`, else
                 K real-world rollouts and K simulated rollouts of equal length
        """
        if not isinstance(rollouts_real[0], Iterable):
            raise pyrado.TypeErr(given=rollouts_real[0], expected_type=Iterable)
        if not isinstance(rollouts_sim[0], Iterable):
            raise pyrado.TypeErr(given=rollouts_sim[0], expected_type=Iterable)
        if not replicate and len(rollouts_real) != len(rollouts_sim):
            raise pyrado.ShapeErr(msg="In case of a one on one comparison, the number of rollouts needs to be equal!")

        # Choose the function for creating the comparison, the rollouts
        comp_fcn = product if replicate else zip

        # Go over all combinations rollouts individually
        rollouts_real_tr = []
        rollouts_sim_tr = []
        for ro_r, ro_s in comp_fcn(rollouts_real, rollouts_sim):
            # Handle rollouts of different length, assuming that they are staring at the same state
            if ro_r.length < ro_s.length:
                rollouts_real_tr.append(ro_r)
                rollouts_sim_tr.append(ro_s[: ro_r.length])
            elif ro_r.length > ro_s.length:
                rollouts_real_tr.append(ro_r[: ro_s.length])
                rollouts_sim_tr.append(ro_s)
            else:
                rollouts_real_tr.append(ro_r)
                rollouts_sim_tr.append(ro_s)

        return rollouts_real_tr, rollouts_sim_tr

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        # ParameterExploring subroutine saves the best policy (in this case a DomainDistrParamPolicy)
        if "prefix" in meta_info:
            self._subrtn.save_snapshot(meta_info=dict(prefix=f"{meta_info['prefix']}_ddp"))  # save iter_X_ddp_policy.pt
        self._subrtn.save_snapshot(meta_info=dict(prefix="ddp"))  # override ddp_policy.pt

        joblib.dump(self._subrtn.env, osp.join(self.save_dir, "env_sim.pkl"))

        # Print the current search distribution's mean
        cpp = self._subrtn.policy.transform_to_ddp_space(self._subrtn.policy.param_values)
        self._subrtn.env.adapt_randomizer(domain_distr_param_values=cpp.detach().cpu().numpy())
        print_cbt(f"Current policy domain parameter distribution\n{self._subrtn.env.randomizer}", "g")

        # Set the randomizer to best fitted domain distribution
        cbp = self._subrtn.policy.transform_to_ddp_space(self._subrtn.best_policy_param)
        self._subrtn.env.adapt_randomizer(domain_distr_param_values=cbp.detach().cpu().numpy())
        print_cbt(f"Best fitted domain parameter distribution\n{self._subrtn.env.randomizer}", "g")

        if "rollouts_real" not in meta_info:
            raise pyrado.KeyErr(keys="rollouts_real", container=meta_info)
        pyrado.save(meta_info["rollouts_real"], "rollouts_real", "pkl", self.save_dir, meta_info)
