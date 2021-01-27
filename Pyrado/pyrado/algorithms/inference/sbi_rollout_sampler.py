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
import torch as to
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Union, Mapping, Optional, Tuple, List

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.spaces.discrete import DiscreteSpace


class RolloutSamplerForSBI(ABC):
    """
    Wrapper to do ebnble the sbi simulator instance to make rollouts from SimuRLacra environments as if the environment
    was a callable that only needs the simulator parameters as inputs
    """

    def __init__(self, env: Env, policy: Policy, strategy: str):
        """
        Constructor

        :param env: environment which the policy operates, in sim-to-real settings this is a real-world device, buy in
                    a sim-to-sim experiment this can be a (randomized) `SimEnv`. We strip all domain randomization
                    wrappers from this env since we want to randomize it manually here.
        :param policy: policy used for sampling the rollout
        :param strategy: method with which the observations are computed from the rollouts. Possible options:
                         `states` (uses all observed states from rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in  [1])

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019
        """
        if not strategy.lower() in ["states", "final_state", "bayessim"]:
            raise pyrado.ValueErr(given=strategy, eq_constraint="states, final_state, bayessim")

        self._env = env
        self._policy = policy
        self.strategy = strategy.lower()

    @abstractmethod
    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        raise NotImplementedError

    def transform_data(self, rollout: StepSequence):
        """
        Transforms rollouts into the observations used for likelihood-free inference.
        Currently a state-representation as well as state-action summary-statistics are available.

        :param rollout: one rollout containing the data which should be transformed into an observation for inference
        :return: observation used for inference
        """
        if self.strategy == "states":
            context_strat = self.all_states
        elif self.strategy == "final_state":
            context_strat = self.final_state
        elif self.strategy == "bayessim":
            context_strat = self.bayessim_statistic
        else:
            raise pyrado.ValueErr(given=self.strategy)

        return context_strat(rollout)

    @staticmethod
    def all_states(rollout: StepSequence) -> to.Tensor:
        """
        Returns the observations of the rollout as a vector.
        Can be used if each trajectory has the same size or for rnn networks.

        :param rollout: one rollout containing the data which should be transformed into an observation for inference
        :return: observations as a vector
        """
        rollout.torch(data_type=to.get_default_dtype())

        return rollout.observations.view(-1)

    @staticmethod
    def final_state(rollout: StepSequence) -> to.Tensor:
        """
        Returns the last observations of the rollout as a vector.

        :param rollout: one rollout containing the data which should be transformed into an observation for inference
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

        :param rollout: one rollout containing the data which should be transformed into an observation for inference
        :return: summary statistics of the rollout
        """
        rollout.torch(data_type=to.get_default_dtype())

        act = rollout.actions
        obs = rollout.observations  # len(obs) = len(act)+1
        obs_diff = obs[1:] - obs[:-1]

        # Compute the statistics
        act_obs_dot_prod = to.einsum("ij,ik->jk", act, obs_diff).view(-1)
        mean_obs_diff = to.mean(obs_diff, dim=0)
        var_obs_diff = to.mean((mean_obs_diff - obs_diff) ** 2, dim=0)

        # Combine all the statistics
        return to.cat((act_obs_dot_prod, mean_obs_diff, var_obs_diff), dim=0)


class SimRolloutSamplerForSBI(RolloutSamplerForSBI):
    """ Wrapper to make SimuRLacra's simulation environments usable as simulators for the sbi package """

    def __init__(
        self,
        env: Union[SimEnv, EnvWrapper],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        strategy: str,
        rollouts_real: Optional[List[StepSequence]] = None,
    ):
        """
        Constructor

        :param env: environment which the policy operates, in any case this will be a (randomized) `SimEnv`. We strip
                    all domain randomization wrappers from this env since we want to randomize it manually here.
        :param policy: policy used for sampling the rollout
        :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass)
        :param strategy: the method with which the observations are computed from the rollouts. Possible options:
                         `states` (uses all observed states from rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in  [1])
        :param rollouts_real: list of rollouts recorded from the real system, which are used to sync the simulations'
                              initial states

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

        super().__init__(env=env, policy=policy, strategy=strategy)

        self.dp_names = dp_mapping.values()
        self.rollouts_real = rollouts_real

    def __call__(self, dp_values: to.Tensor) -> to.Tensor:
        """
        Set the domain parameter, run one rollout, and compute summary statistics.

        :param dp_values: tensor containing the domain parameter values [num samples x num domain parameters]

        .. note::
            If it is not desired that sbi treats this function as a batched simulator, just insert

            .. code-block:: python

                    if dp_values.ndim == 2:
                        raise RuntimeError
        """
        dp_values = to.atleast_2d(dp_values)

        # Set the initial state space during __call__() otherwise it gets magically set back to the default
        if self.rollouts_real is not None:
            init_states_real = np.stack([ro.rollout_info["init_state"] for ro in self.rollouts_real])
            if not init_states_real.shape == (len(self.rollouts_real), self._env.state_space.flat_dim):
                raise pyrado.ShapeErr(
                    given=init_states_real, expected_match=(len(self.rollouts_real), self._env.state_space.flat_dim)
                )
            self._env.init_space = DiscreteSpace(init_states_real)

        # Do the rollouts
        ros = [
            rollout(
                self._env,
                self._policy,
                eval=True,
                reset_kwargs=dict(domain_param=dict(zip(self.dp_names, dpv))),
            )
            for dpv in dp_values.numpy()
        ]

        # Check if the domain parameters in the rollout are actually the ones commanded by sbi
        if not all(
            [
                to.allclose(to.as_tensor(itemgetter(*self.dp_names)(ro.rollout_info["domain_param"])), dpv)
                for ro, dpv in zip(ros, dp_values)
            ]
        ):
            raise pyrado.ValueErr(
                msg="The domain parameters after the rollouts are not identical to the ones commanded by the sbi!"
            )

        # Transform the data to torch and compute the observations used for inference from the rollout data
        obs_real = to.stack([self.transform_data(ro) for ro in ros])

        if obs_real.shape[0] != dp_values.shape[0]:
            raise pyrado.ShapeErr(given=obs_real, expected_match=dp_values)

        return obs_real


class RealRolloutSamplerForSBI(RolloutSamplerForSBI):
    """ Wrapper to make SimuRLacra's real environments similar to the simulators for the sbi package """

    def __init__(
        self,
        env: Env,
        policy: Policy,
        strategy: str,
    ):
        """
        Constructor

        :param env: environment which the policy operates, in sim-to-real settings this is a real-world device, buy in
                    a sim-to-sim experiment this can be a (randomized) `SimEnv`. We strip all domain randomization
                    wrappers from this env since we want to randomize it manually here.
        :param policy: policy used for sampling the rollout
        :param strategy: the method with which the observations are computed from the rollouts. Possible options:
                         `states` (uses all observed states from rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `bayessim` (summary statistics as proposed in  [1])

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", arXiv, 2019
        """
        super().__init__(env=env, policy=policy, strategy=strategy)

    def __call__(self, dp_values: to.Tensor = None) -> Tuple[to.Tensor, StepSequence]:
        r"""
        Run one rollout and compute summary statistics.

        :param dp_values: ignored, just here for the interface compatibility
        :return: observation a.k.a. $x_o$, and initial state of the physical device
        """
        # Don't set the domain params here since they are set by the DomainRandWrapperBuffer to mimic the randomness
        ro = rollout(self._env, self._policy, eval=True)

        # Return the observations used for inference from the rollout data
        obs_real = self.transform_data(ro)

        return obs_real, ro
