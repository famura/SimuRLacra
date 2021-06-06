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
from abc import ABC, abstractmethod
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.policies.feed_forward.playback import PlaybackPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sbi_embeddings import Embedding
from pyrado.sampling.step_sequence import StepSequence, check_act_equal
from pyrado.spaces import BoxSpace
from pyrado.utils.checks import check_all_lengths_equal
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt_once


class RolloutSamplerForSBI(ABC, Serializable):
    """
    Wrapper to do enable the sbi simulator instance to make rollouts from SimuRLacra environments as if the environment
    was a callable that only needs the simulator parameters as inputs

    .. note::
        The features of each rollout are concatenated, and since the inference procedure requires a consistent size of
        the inputs, it is necessary that all rollouts yield the same number of features, i.e. have equal length!
    """

    def __init__(
        self, env: Env, policy: Policy, embedding: Embedding, num_segments: int = None, len_segments: int = None
    ):
        """
        Constructor

        :param env: environment which the policy operates, in sim-to-real settings this is a real-world device, buy in
                    a sim-to-sim experiment this can be a (randomized) `SimEnv`. We strip all domain randomization
                    wrappers from this env since we want to randomize it manually here.
        :param policy: policy used for sampling the rollout
        :param embedding: embedding used for pre-processing the data before (later) passing it to the posterior
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        """
        if num_segments is None and len_segments is None or num_segments is not None and len_segments is not None:
            raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")

        Serializable._init(self, locals())

        self._env = env
        self._policy = policy
        self.num_segments = num_segments
        self.len_segments = len_segments
        self._embedding = embedding

    @abstractmethod
    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        raise NotImplementedError

    @staticmethod
    def get_dim_data(spec: EnvSpec) -> int:
        """
        Compute the dimension of the data which is extracted from the rollouts.

        :param spec: environment specification
        :return: dimension of one data sample, i.e. one time step
        """
        return spec.state_space.flat_dim + spec.act_space.flat_dim


class SimRolloutSamplerForSBI(RolloutSamplerForSBI, Serializable):
    """Wrapper to make SimuRLacra's simulation environments usable as simulators for the sbi package"""

    def __init__(
        self,
        env: Union[SimEnv, EnvWrapper],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        embedding: Embedding,
        num_segments: int = None,
        len_segments: int = None,
        rollouts_real: Optional[List[StepSequence]] = None,
        use_rec_act: bool = True,
    ):
        """
        Constructor

        :param env: environment which the policy operates, which must not be a randomized environment since we want to
                    randomize it manually via the domain parameters coming from the sbi package
        :param policy: policy used for sampling the rollout
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param embedding: embedding used for pre-processing the data before (later) passing it to the posterior
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        :param use_rec_act: if `True` the recorded actions form the target domain are used to generate the rollout
                            during simulation (feed-forward). If `False` there policy is used to generate (potentially)
                            state-dependent actions (feed-back).
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
            env=env, policy=policy, embedding=embedding, num_segments=num_segments, len_segments=len_segments
        )

        self.dp_names = dp_mapping.values()
        self.rollouts_real = rollouts_real
        self.use_rec_act = use_rec_act

    def __call__(self, dp_values: to.Tensor) -> to.Tensor:
        """
        Run one rollout for every domain parameter set. The rollouts are done in segments, and after every segment the
        simulation state is set to the current state in the target domain rollout.

        :param dp_values: tensor containing domain parameters along the 1st dimension
        :return: features computed from the time series data
        """
        dp_values = to.atleast_2d(dp_values).numpy()

        if self.rollouts_real is not None:
            if self.use_rec_act:
                # Create a policy that simply replays the recorded actions
                policy = PlaybackPolicy(
                    self._env.spec, [ro.actions_applied for ro in self.rollouts_real], no_reset=True
                )
            else:
                # Use the current policy to generate the actions
                policy = self._policy

            # The initial states will be set to states which will most likely not the be in the initial state space of
            # the environment, thus we set the initial state space to an infinite space
            self._env.init_space = BoxSpace(
                -pyrado.inf, pyrado.inf, self._env.state_space.shape, labels=self._env.state_space.labels
            )

            data_sim_all = []  # for all target domain rollouts

            # Iterate over domain parameter sets
            for dp_value in dp_values:
                data_sim_one_dp = []  # for all target domain rollouts of one domain parameter set

                # Iterate over target domain rollouts
                for idx_r, ro_real in enumerate(self.rollouts_real):
                    data_one_ro = []
                    ro_real.numpy()

                    # Split the target domain rollout if desired
                    if self.num_segments is not None:
                        segs_real = list(ro_real.split_ordered_batches(num_batches=self.num_segments))
                    else:
                        segs_real = list(ro_real.split_ordered_batches(batch_size=self.len_segments))

                    # Iterate over segments of one target domain rollout
                    cnt_step = 0
                    for seg_real in segs_real:
                        if self.use_rec_act:
                            # Disabled the policy reset of PlaybackPolicy to do it here manually
                            assert policy.no_reset
                            policy.curr_rec = idx_r
                            policy.curr_step = cnt_step

                        # Do the rollout for a segment
                        seg_sim = rollout(
                            self._env,
                            policy,
                            eval=True,
                            reset_kwargs=dict(
                                init_state=seg_real.states[0, :], domain_param=dict(zip(self.dp_names, dp_value))
                            ),
                            stop_on_done=False,
                            max_steps=seg_real.length,
                        )
                        # check_domain_params(seg_sim, dp_value, self.dp_names)
                        if self.use_rec_act:
                            check_act_equal(seg_real, seg_sim, check_applied=True)

                        # Increase step counter for next segment
                        cnt_step += seg_real.length

                        # Concatenate states and actions of the simulated and real segments
                        data_one_seg = np.concatenate(
                            [seg_sim.states[: len(seg_real), :], seg_sim.actions_applied[: len(seg_real), :]], axis=1
                        )
                        if self._embedding.requires_target_domain_data:
                            # The embedding is also using target domain data (the case for DTW distance)
                            data_one_seg_real = np.concatenate(
                                [seg_real.states[: len(seg_real), :], seg_real.actions_applied], axis=1
                            )
                            data_one_seg = np.concatenate([data_one_seg, data_one_seg_real], axis=1)
                        data_one_seg = to.from_numpy(data_one_seg).to(dtype=to.get_default_dtype())
                        data_one_ro.append(data_one_seg)

                    # Append one simulated rollout
                    data_sim_one_dp.append(to.cat(data_one_ro, dim=0))

                # Append the segments of all target domain rollouts for the current domain parameter
                data_sim_all.append(to.stack(data_sim_one_dp, dim=0))

            # Compute the features from all time series
            data_sim_all = to.stack(data_sim_all, dim=0)  # shape [batch_size, num_rollouts, len_time_series, dim_data]
            data_sim_all = self._embedding(Embedding.pack(data_sim_all))  # shape [batch_size, num_rollouts * dim_data]

            # Check shape
            if data_sim_all.shape != (dp_values.shape[0], len(self.rollouts_real) * self._embedding.dim_output):
                raise pyrado.ShapeErr(
                    given=data_sim_all,
                    expected_match=(dp_values.shape[0], len(self.rollouts_real) * self._embedding.dim_output),
                )

        else:
            # There are no pre-recorded rollouts, e.g. during _setup_sbi().
            # Use the current policy yo generate the actions.
            policy = self._policy

            # Do the rollouts
            data_sim_all = []
            for dp_value in dp_values:
                ro_sim = rollout(
                    self._env,
                    policy,
                    eval=True,
                    reset_kwargs=dict(domain_param=dict(zip(self.dp_names, dp_value))),
                    stop_on_done=False,
                )
                # check_domain_params(ro_sim, dp_value, self.dp_names)

                # Concatenate states and actions of the simulated segments
                data_one_seg = np.concatenate([ro_sim.states[:-1, :], ro_sim.actions_applied], axis=1)
                if self._embedding.requires_target_domain_data:
                    data_one_seg = np.concatenate([data_one_seg, data_one_seg], axis=1)
                data_one_seg = to.from_numpy(data_one_seg).to(dtype=to.get_default_dtype())
                data_sim_all.append(data_one_seg)

            # Compute the features from all time series
            data_sim_all = to.stack(data_sim_all, dim=0)
            data_sim_all = data_sim_all.unsqueeze(1)  # equivalent to only one target domain rollout
            data_sim_all = self._embedding(Embedding.pack(data_sim_all))  # shape [batch_size,  dim_feat]

            # Check shape
            if data_sim_all.shape != (dp_values.shape[0], self._embedding.dim_output):
                raise pyrado.ShapeErr(
                    given=data_sim_all, expected_match=(dp_values.shape[0], self._embedding.dim_output)
                )

        return data_sim_all  # shape [batch_size, num_rollouts * dim_feat]


class RealRolloutSamplerForSBI(RolloutSamplerForSBI, Serializable):
    """Wrapper to make SimuRLacra's real environments similar to the sbi simulator"""

    def __init__(
        self,
        env: Env,
        policy: Policy,
        embedding: Embedding,
        num_segments: int = None,
        len_segments: int = None,
    ):
        """
        Constructor

        :param env: environment which the policy operates, in sim-to-real settings this is a real-world device, i.e.
                    `RealEnv`, but in a sim-to-sim experiment this can be a (randomized) `SimEnv`
        :param policy: policy used for sampling the rollout
        :param embedding: embedding used for pre-processing the data before (later) passing it to the posterior
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        """

        Serializable._init(self, locals())

        super().__init__(
            env=env, policy=policy, embedding=embedding, num_segments=num_segments, len_segments=len_segments
        )

    def __call__(self, dp_values: to.Tensor = None) -> Tuple[to.Tensor, StepSequence]:
        """
        Run one rollout in the target domain, and compute the features of the data used for sbi.

        :param dp_values: ignored, just here for the interface compatibility
        :return: features computed from the time series data, and the complete rollout
        """
        ro = None
        run_interactive_loop = True
        while run_interactive_loop:
            # Don't set the domain params here since they are set by the DomainRandWrapperBuffer to mimic the randomness
            ro = rollout(self._env, self._policy, eval=True, stop_on_done=False)
            if not isinstance(self._env, RealEnv):
                run_interactive_loop = False
            else:
                # Ask is the current rollout should be discarded and redone
                run_interactive_loop = input("Continue with the next rollout y / n? ").lower() == "n"
        ro.torch()

        data_real = to.cat([ro.states[:-1, :], ro.actions_applied], dim=1)
        if self._embedding.requires_target_domain_data:
            data_real = to.cat([data_real, data_real], dim=1)

        # Compute the features
        data_real = data_real.unsqueeze(0)  # only one target domain rollout
        data_real = self._embedding(Embedding.pack(data_real))  # shape [1, dim_feat]

        # Check shape (here no batching and always one rollout)
        if data_real.shape[0] != 1 or data_real.ndim != 2:
            raise pyrado.ShapeErr(given=data_real, expected_match=(1, -1))

        return data_real, ro


class RecRolloutSamplerForSBI(RealRolloutSamplerForSBI, Serializable):
    """Wrapper to yield pre-recorded rollouts similar to the sbi simulator"""

    def __init__(
        self,
        rollouts_dir: str,
        embedding: Embedding,
        num_segments: int = None,
        len_segments: int = None,
        rand_init_rollout: bool = True,
    ):
        """
        Constructor

        :param rollouts_dir: directory where to find the of pre-recorded rollouts
        :param num_segments: number of segments in which the rollouts are split into. For every segment, the initial
                             state of the simulation is reset, and thus for every set the features of the trajectories
                             are computed separately. Either specify `num_segments` or `len_segments`.
        :param embedding: embedding used for pre-processing the data before (later) passing it to the posterior
        :param len_segments: length of the segments in which the rollouts are split into. For every segment, the initial
                            state of the simulation is reset, and thus for every set the features of the trajectories
                            are computed separately. Either specify `num_segments` or `len_segments`.
        :param rand_init_rollout: if `True`, chose the first rollout at random, and then cycle through the list
        """
        if not os.path.isdir(rollouts_dir):
            raise pyrado.PathErr(given=rollouts_dir)

        Serializable._init(self, locals())

        super().__init__(
            env=None, policy=None, embedding=embedding, num_segments=num_segments, len_segments=len_segments
        )

        # Crawl through the directory and load every file that starts with the word rollout
        rollouts_rec = []
        for root, dirs, files in os.walk(rollouts_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            rollouts_rec = [pyrado.load(name=f, load_dir=root) for f in files if f.startswith("rollout")]
            check_all_lengths_equal(rollouts_rec)
        if not rollouts_rec:
            raise pyrado.ValueErr(msg="No rollouts have been found!")

        self.rollouts_dir = rollouts_dir
        self.rollouts_rec = rollouts_rec
        self._ring_idx = np.random.randint(0, len(rollouts_rec)) if rand_init_rollout else 0

    @property
    def ring_idx(self) -> int:
        """Get the buffer's index."""
        return self._ring_idx

    @ring_idx.setter
    def ring_idx(self, idx: int):
        """Set the buffer's index."""
        if not (isinstance(idx, int) or not 0 <= idx < self.num_rollouts):
            raise pyrado.ValueErr(given=idx, ge_constraint="0 (int)", l_constraint=self.num_rollouts)
        self._ring_idx = idx

    @property
    def num_rollouts(self) -> int:
        """Get the number of stored rollouts."""
        return len(self.rollouts_rec)

    def __call__(self, dp_values: to.Tensor = None) -> Tuple[to.Tensor, StepSequence]:
        """
        Yield one rollout from the pre-recorded buffer of rollouts, and compute the features of the data used for sbi.

        :param dp_values: ignored, just here for the interface compatibility
        :return: features computed from the time series data, and the complete rollout
        """
        print_cbt_once(f"Using pre-recorded target domain rollouts to from {self.rollouts_dir}", "g")

        # Get pre-recoded rollout and advance the index
        if not isinstance(self.rollouts_rec, list) and isinstance(self.rollouts_rec[0], StepSequence):
            raise pyrado.TypeErr(given=self.rollouts_rec, expected_type=list)

        ro = self.rollouts_rec[self._ring_idx]
        ro.torch()
        self._ring_idx = (self._ring_idx + 1) % self.num_rollouts

        data_real = to.cat([ro.states[:-1, :], ro.actions_applied], dim=1)
        if self._embedding.requires_target_domain_data:
            data_real = to.cat([data_real, data_real], dim=1)

        # Compute the features
        data_real = data_real.unsqueeze(0)  # only one target domain rollout
        data_real = self._embedding(Embedding.pack(data_real))  # shape [1, dim_feat]

        # Check shape (here no batching and always one rollout)
        if data_real.shape[0] != 1 or data_real.ndim != 2:
            raise pyrado.ShapeErr(given=data_real, expected_match=(1, -1))

        return data_real, ro
