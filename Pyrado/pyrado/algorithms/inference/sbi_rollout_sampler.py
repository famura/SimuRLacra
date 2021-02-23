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
import os
import torch as to
from abc import ABC, abstractmethod
from init_args_serializer import Serializable
from operator import itemgetter
from typing import Union, Mapping, Optional, Tuple, List, ValuesView

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.policies.special.time import PlaybackPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces import BoxSpace
from pyrado.utils.checks import check_act_equal
from pyrado.utils.input_output import print_cbt_once


def _check_domain_params(
    rollouts: List[StepSequence], domain_param_value: np.ndarray, domain_param_names: Union[List[str], ValuesView]
):
    """
    Verify if the domain parameters in the rollout are actually the ones commanded.

    :param rollouts: simulated rollouts or rollout segments
    :param domain_param_value: one set of domain parameters as commanded
    :param domain_param_names: names of the domain parameters to set, i.e. values of the domain parameter mapping
    """
    if not all(
        [
            np.allclose(
                np.asarray(itemgetter(*domain_param_names)(ro.rollout_info["domain_param"])), domain_param_value
            )
            for ro in rollouts
        ]
    ):
        raise pyrado.ValueErr(
            msg="The domain parameters after the rollouts are not identical to the ones commanded by the sbi!"
        )


class RolloutSamplerForSBI(ABC, Serializable):
    """
    Wrapper to do enable the sbi simulator instance to make rollouts from SimuRLacra environments as if the environment
    was a callable that only needs the simulator parameters as inputs
    """

    def __init__(self, env: Env, policy: Policy, strategy: str, num_segments: int = None, len_segments: int = None):
        """
        Constructor

        :param env: environment which the policy operates, in sim-to-real settings this is a real-world device, buy in
                    a sim-to-sim experiment this can be a (randomized) `SimEnv`. We strip all domain randomization
                    wrappers from this env since we want to randomize it manually here.
        :param policy: policy used for sampling the rollout
        :param strategy: method with which the observations are computed from the rollouts. Possible options:
                         `dtw_distance` (dynamic time warping using all observations from the rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in [1])
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019
        """
        if not strategy.lower() in ["dtw_distance", "final_state", "bayessim"]:
            raise pyrado.ValueErr(given=strategy, eq_constraint="dtw_distance, final_state, bayessim")

        Serializable._init(self, locals())

        self._env = env
        self._policy = policy
        self.strategy = strategy.lower()
        if num_segments is None and len_segments is None or num_segments is not None and len_segments is not None:
            raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")
        self.num_segments = num_segments
        self.len_segments = len_segments

    @abstractmethod
    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        raise NotImplementedError

    @property
    def dim_output(self):
        """ Get the output dimension of the respective transformation. """
        if self.strategy == "dtw_distance":
            d = 1
        elif self.strategy == "final_state":
            d = self._env.obs_space.shape[0]
        elif self.strategy == "bayessim":
            obs_dim = self._env.obs_space.shape[0]
            act_dim = self._env.act_space.shape[0]
            d = obs_dim * act_dim + 2 * obs_dim
        else:
            raise NotImplementedError

        return d * self.num_segments if self.num_segments is not None else d

    def compute_observations(self, rollout: StepSequence) -> to.Tensor:
        """
        Compute the observations from a given rollout, depending on the transformation and the way to segment the
        rollout. In that process, the last segment might be ignored to get rid off too short segments.

        :param rollout: input rollout data
        :return: feature values
        """
        if self.num_segments is not None:
            segments = list(rollout.split_ordered_batches(num_batches=self.num_segments + 1))
            segments = segments[:-1]

            # Transform the data to torch and compute the observations used for inference from the rollout data.
            # This is done for all segments separately, and since we know how many segments there will be, we can
            # concatenate them, and use them individually
            obs_real = to.cat([self.transform_data(seg, None) for seg in segments], dim=0)

        else:
            segments = list(rollout.split_ordered_batches(batch_size=self.len_segments))
            if segments[-1].length < 2:
                segments = segments[:-1]

            # Transform the data to torch and compute the observations used for inference from the rollout data.
            # This is done for all segments separately, and since we know don't how many segments there will be,
            # we can only average them
            obs_real = to.mean(to.stack([self.transform_data(seg, None) for seg in segments]), dim=0)

        return obs_real

    def transform_data(self, rollout_query: StepSequence, rollouts_ref: Optional[List[StepSequence]]) -> to.Tensor:
        r"""
        Transforms rollouts into the observations used for likelihood-free inference.
        Currently a state-representation as well as state-action summary-statistics are available.

        :param rollout_query: rollout or segment thereof containing the data to be transformed for inference
        :param rollouts_ref: reference rollout(s) from the target domain, if `None` the reference is set to the the
                             query. The latter case is true for computing the statistics for the target domain rollouts
        :return: observation used for inference, a.k.a $x_o$
        """
        if self.strategy == "dtw_distance":
            return self.dtw_distance(rollout_query, rollouts_ref)
        elif self.strategy == "final_state":
            return self.final_state(rollout_query)
        elif self.strategy == "bayessim":
            assert rollout_query.length > 1
            return self.bayessim_statistic(rollout_query)
        else:
            raise pyrado.ValueErr(given=self.strategy)

    @staticmethod
    def dtw_distance(
        rollout_query: StepSequence, rollouts_ref: Optional[Union[StepSequence, List[StepSequence]]]
    ) -> to.Tensor:
        """
        Returns the dynamic time warping distance between the rollouts' observations.

        .. note::
            It is necessary to take the mean over all distances since the same function is used to compute the
            observations (for sbi) form the target domain rollouts. At this point in time there might be only one target
            domain rollout, thus the target domain rollouts are only compared with themselves, thus yield a scalar
            distance value.

        :param rollout_query: rollout or segment thereof containing the data to be transformed for inference
        :param rollouts_ref: reference rollout(s) from the target domain, if `None` the reference is set to the the
                             query. The latter case is true for computing the statistics for the target domain rollouts
        :return: dynamic time warping distance in multi-dim observations space, averaged over target domain rollouts
        """
        from dtw import dtw

        if rollouts_ref is None:
            rollouts_ref = [rollout_query]
        elif isinstance(rollouts_ref, StepSequence):
            rollouts_ref = [rollouts_ref]
        if not isinstance(rollouts_ref, (StepSequence, list)):
            raise pyrado.TypeErr(given=rollouts_ref, expected_type=(StepSequence, list))
        if isinstance(rollouts_ref, list) and not isinstance(rollouts_ref[0], StepSequence):
            raise pyrado.TypeErr(given=rollouts_ref[0], expected_type=StepSequence)

        # Align the rollouts with the Rabiner-Juang type VI-c unsmoothed recursion
        distances = []
        for ro_ref in rollouts_ref:
            distances.append(
                dtw(
                    rollout_query.observations,
                    ro_ref.observations,
                    open_end=True,
                    # step_pattern=rabinerJuangStepPattern(6, "c"),
                ).distance
            )

        return to.mean(to.as_tensor(distances, dtype=to.get_default_dtype())).view(1)

    @staticmethod
    def final_state(rollout: StepSequence) -> to.Tensor:
        """
        Returns the last observations of the rollout as a vector.

        :param rollout: rollout or segment thereof containing the data to be transformed for inference
        :return: last observations as a vector
        """
        rollout.torch(data_type=to.get_default_dtype())

        return rollout.observations[-1].view(-1)

    @staticmethod
    def bayessim_statistic(rollout: StepSequence) -> to.Tensor:
        """
        Computing summary statistics based on approach in [1], see eq. (22).
        This method guarantees output which has the same size for every trajectory.

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019

        :param rollout: rollout or segment thereof containing the data to be transformed for inference
        :return: summary statistics of the rollout
        """
        if rollout.length < 2:
            raise pyrado.ShapeErr(given=rollout, expected_match=(2, -1))

        rollout.torch(data_type=to.get_default_dtype())
        act = rollout.actions if len(rollout.observations) == len(rollout.actions) + 1 else rollout.actions[:-1]
        obs = rollout.observations  #
        obs_diff = obs[1:] - obs[:-1]

        # Compute the statistics
        act_obs_dot_prod = to.einsum("ij,ik->jk", act, obs_diff).view(-1)
        mean_obs_diff = to.mean(obs_diff, dim=0)
        var_obs_diff = to.mean((mean_obs_diff - obs_diff) ** 2, dim=0)

        # Combine all the statistics
        return to.cat((act_obs_dot_prod, mean_obs_diff, var_obs_diff), dim=0)


class SimRolloutSamplerForSBI(RolloutSamplerForSBI, Serializable):
    """ Wrapper to make SimuRLacra's simulation environments usable as simulators for the sbi package """

    def __init__(
        self,
        env: Union[SimEnv, EnvWrapper],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        strategy: str,
        num_segments: int = None,
        len_segments: int = None,
        rollouts_real: Optional[List[StepSequence]] = None,
    ):
        """
        Constructor

        :param env: environment which the policy operates, which must not be a randomized environment since we want to
                    randomize it manually via the domain parameters coming from the sbi package
        :param policy: policy used for sampling the rollout
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        :param strategy: the method with which the observations are computed from the rollouts. Possible options:
                         `dtw_distance` (dynamic time warping using all observations from the rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in [1])

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019
        """
        if typed_env(env, DomainRandWrapper):
            raise pyrado.TypeErr(
                msg="The environment passed to sbi as simulator must not be wrapped with a subclass of"
                "DomainRandWrapper since sbi has be able to set the domain parameters explicitly!"
            )
        if rollouts_real is not None:
            if not isinstance(rollouts_real, list):
                raise pyrado.TypeErr(given=rollouts_real, expected_type=list)
            if not isinstance(rollouts_real[0], StepSequence):  # only check 1st element
                raise pyrado.TypeErr(given=rollouts_real[0], expected_type=StepSequence)

        Serializable._init(self, locals())

        super().__init__(
            env=env, policy=policy, strategy=strategy, num_segments=num_segments, len_segments=len_segments
        )

        self.dp_names = dp_mapping.values()
        self.rollouts_real = rollouts_real

    def __call__(self, dp_values: to.Tensor) -> to.Tensor:
        """
        Set the domain parameter, run one rollout, and compute summary statistics.

        :param dp_values: tensor containing the domain parameter values [num samples x num domain parameters]
        """
        dp_values = to.atleast_2d(dp_values).numpy()

        if self.rollouts_real is not None:
            # Create a policy that simply replays the recorded actions
            policy = PlaybackPolicy(self._env.spec, [ro.actions for ro in self.rollouts_real], no_reset=True)

            # The initial states will be set to states which will most likely not the be in the initial state space of
            # the environment, thus we set the initial state space to an infinite space
            self._env.init_space = BoxSpace(
                -pyrado.inf, pyrado.inf, self._env.state_space.shape, labels=self._env.state_space.labels
            )

            obs_real_all = []  # for all target domain rollouts

            # Iterate over domain parameter sets
            for dp_value in dp_values:
                obs_real_one_dp = []  # for all target domain rollouts of one domain parameter set

                # Iterate over target domain rollouts
                for idx_r, ro_real in enumerate(self.rollouts_real):
                    ro_real.numpy()
                    # Split the target domain rollout, see compute_observations()
                    if self.num_segments is not None:
                        segs_real = list(ro_real.split_ordered_batches(num_batches=self.num_segments + 1))
                        segs_real = segs_real[:-1]
                    else:
                        segs_real = list(ro_real.split_ordered_batches(batch_size=self.len_segments))
                        if segs_real[-1].length < 2:
                            segs_real = segs_real[:-1]

                    segs_sim = []
                    cnt_step = 0

                    # Iterate over segments of one target domain rollout
                    for seg_real in segs_real:
                        # Disabled the policy reset of PlaybackPolicy to do it here manually
                        policy.curr_rec = idx_r
                        policy.curr_step = cnt_step

                        # Do the rollout for a segment
                        seg_sim = rollout(
                            self._env,
                            policy,
                            eval=True,
                            reset_kwargs=dict(
                                init_state=seg_real.states[0], domain_param=dict(zip(self.dp_names, dp_value))
                            ),
                            max_steps=seg_real.length,
                        )
                        check_act_equal(seg_real, seg_sim)

                        # Append the current segment, and increase step counter for next segment
                        segs_sim.append(seg_sim)
                        cnt_step += seg_real.length

                    _check_domain_params(segs_sim, dp_value, self.dp_names)
                    assert len(segs_sim) == len(segs_real)

                    # Compute the observations, see compute_observations()
                    if self.num_segments is not None:
                        obs_real_segs = to.cat(
                            [self.transform_data(s_sim, s_real) for s_sim, s_real in zip(segs_sim, segs_real)], dim=0
                        )
                    else:
                        obs_real_segs = to.mean(
                            to.stack(
                                [self.transform_data(s_sim, s_real) for s_sim, s_real in zip(segs_sim, segs_real)]
                            ),
                            dim=0,
                        )
                    obs_real_one_dp.append(obs_real_segs)

                # Append the mean observation, averaged over target domain rollouts
                obs_real_all.append(to.mean(to.stack(obs_real_one_dp), dim=0))

        else:
            # There are no pre-recorded rollouts, e.g. during _setup_sbi() in LFI.__init__()
            policy = self._policy

            # Do the rollouts
            obs_real_all = []
            for dpv in dp_values:
                ro_sim = rollout(
                    self._env,
                    policy,
                    eval=True,
                    reset_kwargs=dict(domain_param=dict(zip(self.dp_names, dpv))),
                )
                # Get the observations from the simulated rollout
                obs_real_segs = self.compute_observations(ro_sim)
                obs_real_all.append(obs_real_segs)

        # Stack and check
        obs_real_all = to.stack(obs_real_all, dim=0)
        if obs_real_all.shape[0] != dp_values.shape[0]:
            raise pyrado.ShapeErr(given=obs_real_all, expected_match=dp_values)

        return obs_real_all


class RealRolloutSamplerForSBI(RolloutSamplerForSBI, Serializable):
    """ Wrapper to make SimuRLacra's real environments similar to the sbi simulator """

    def __init__(
        self,
        env: Env,
        policy: Policy,
        strategy: str,
        num_segments: int = None,
        len_segments: int = None,
    ):
        """
        Constructor

        :param env: environment which the policy operates, in sim-to-real settings this is a real-world device, i.e.
                    `RealEnv`, but in a sim-to-sim experiment this can be a (randomized) `SimEnv`
        :param policy: policy used for sampling the rollout
        :param strategy: the method with which the observations are computed from the rollouts. Possible options:
                         `dtw_distance` (dynamic time warping using all observations from the rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in [1])
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019
        """

        Serializable._init(self, locals())

        super().__init__(
            env=env, policy=policy, strategy=strategy, num_segments=num_segments, len_segments=len_segments
        )

    def __call__(self, dp_values: to.Tensor = None) -> Tuple[to.Tensor, StepSequence]:
        r"""
        Run one rollout and compute summary statistics.

        :param dp_values: ignored, just here for the interface compatibility
        :return: observation a.k.a. $x_o$, and initial state of the physical device
        """
        # Don't set the domain params here since they are set by the DomainRandWrapperBuffer to mimic the randomness
        ro = rollout(self._env, self._policy, eval=True)

        obs_real = self.compute_observations(ro)

        return obs_real, ro


class RecRolloutSamplerForSBI(RealRolloutSamplerForSBI, Serializable):
    """ Wrapper to yield pre-recorded rollouts similar to the sbi simulator """

    def __init__(
        self,
        strategy: str,
        rollouts_dir: str,
        num_segments: int = None,
        len_segments: int = None,
        rand_init_rollout: Optional[bool] = True,
    ):
        """
        Constructor

        :param strategy: the method with which the observations are computed from the rollouts. Possible options:
                         `dtw_distance` (dynamic time warping using all observations from the rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in [1])
        :param rollouts_dir: directory where to find the of pre-recorded rollouts
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        :param rand_init_rollout: if `True`, chose the first rollout at random, and then cycle through the list

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019
        """
        if not os.path.isdir(rollouts_dir):
            raise pyrado.PathErr(given=rollouts_dir)

        Serializable._init(self, locals())

        super().__init__(env=None, policy=None, strategy=strategy, num_segments=num_segments, len_segments=len_segments)

        # Crawl through the directory and load every file that starts with the word rollout
        rollouts_rec = []
        for root, dirs, files in os.walk(rollouts_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            rollouts_rec = [
                pyrado.load(None, name=f[: f.rfind(".")], file_ext=f[f.rfind(".") + 1 :], load_dir=root)
                for f in files
                if f.startswith("rollout")
            ]
        if not rollouts_rec:
            raise pyrado.ValueErr(msg="No rollouts have been found!")

        self.rollouts_dir = rollouts_dir
        self.rollouts_rec = rollouts_rec
        self._ring_idx = np.random.randint(0, len(rollouts_rec)) if rand_init_rollout else 0

    @property
    def ring_idx(self) -> int:
        """ Get the buffer's index. """
        return self._ring_idx

    @ring_idx.setter
    def ring_idx(self, idx: int):
        """ Set the buffer's index. """
        if not (isinstance(idx, int) or not 0 <= idx < len(self.rollouts_rec)):
            raise pyrado.ValueErr(given=idx, ge_constraint="0 (int)", l_constraint=len(self.rollouts_rec))
        self._ring_idx = idx

    def __call__(self, dp_values: to.Tensor = None) -> Tuple[to.Tensor, StepSequence]:
        r"""
        Run one rollout and compute summary statistics.

        :param dp_values: ignored, just here for the interface compatibility
        :return: observation a.k.a. $x_o$, and initial state of the physical device
        """
        print_cbt_once(f"Using pre-recorded target domain rollouts to from {self.rollouts_dir}", "g")

        # Get pre-recoded rollout and advance the index
        ro = self.rollouts_rec[self._ring_idx]
        self._ring_idx = (self._ring_idx + 1) % len(self.rollouts_rec)

        obs_real = self.compute_observations(ro)

        return obs_real, ro
