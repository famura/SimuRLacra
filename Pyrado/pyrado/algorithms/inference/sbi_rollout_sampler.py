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

import torch as to
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Mapping

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence


class RolloutSamplerForSBI(ABC):
    """
    Wrapper to do ebnble the sbi simulator instance to make rollouts from SimuRLacra environments as if the environment
    was a callable that only needs the simulator parameters as inputs
    """

    def __init__(self, strategy: str):
        """
        Constructor

        :param strategy: method with which the observations are computed from the rollouts. Possible options:
                         `states` (uses all observed states from rollout),
                         `final_state` (use the last observed state from the rollout), and
                         `ramos` (summary statistics as proposed in  [1])

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", CONFERENCE?, 2020
        """
        if not strategy.lower() in ["states", "final_state", "ramos"]:
            raise pyrado.ValueErr(given=strategy, eq_constraint="states, final_state, summary")

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
        elif self.strategy == "ramos":
            context_strat = self.ramos_statistic
        else:
            raise NotImplementedError

        return context_strat(rollout)

    @staticmethod
    def ramos_statistic(rollout: StepSequence) -> to.Tensor:
        """
        Computing summary statistics based on approach in [1], see eq. (22).
        This method guarantees output which has the same size for every trajectory.

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", CONFERENCE?, 2020

        :param rollout: one rollout containing the data which should be transformed into an observation for inference
        :return: summary statistics of the rollout
        """
        rollout.torch(data_type=to.get_default_dtype())

        act = rollout.actions
        obs = rollout.observations
        # dot product for the state-action dot-product
        obs_diff = obs[1:] - obs[:-1]

        # TODO Sample as below but faster :D
        act_obs_dot_prod = to.einsum("ij,ik->jk", act, obs_diff).view(-1)
        # act_obs_dot_prod = []
        # for a, o in product(act.T, obs_diff.T):
        #     act_obs_dot_prod.append(to.dot(a, o).item())
        # act_obs_dot_prod = to.tensor(act_obs_dot_prod)

        mean_obs_diff = to.mean(obs_diff, dim=0)
        var_obs_diff = to.mean((mean_obs_diff - obs_diff) ** 2, dim=0)

        # Combine all the statistics
        return to.cat((act_obs_dot_prod, mean_obs_diff, var_obs_diff), dim=0)

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


class SimRolloutSamplerForSBI(RolloutSamplerForSBI):
    """ Wrapper to make SimuRLacra's simulation environments usable as simulators for the sbi package """

    def __init__(
        self,
        env: Union[SimEnv, EnvWrapper],
        policy: Policy,
        dp_mapping: Mapping[int, str],
        strategy: str,
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
                         `ramos` (summary statistics as proposed in  [1])
        """
        super().__init__(strategy=strategy)

        self._env = remove_all_dr_wrappers(deepcopy(env))
        self._policy = policy
        self.dp_names = dp_mapping.values()

    def __call__(self, dp_values: to.Tensor):
        """ Set the domain parameter, run one rollout, and compute summary statistics. """
        ro = rollout(
            self._env,
            self._policy,
            eval=True,
            reset_kwargs=dict(domain_param=dict(zip(self.dp_names, dp_values.numpy().flatten()))),
        )
        ro.torch(data_type=to.get_default_dtype())

        # Return the observations used for inference from the rollout data
        return self.transform_data(ro)


class RealRolloutSamplerForSBI(RolloutSamplerForSBI):
    """ Wrapper to make SimuRLacra's real environments usable as simulators for the sbi package """

    def __init__(
        self,
        env: Union[RealEnv, SimEnv, EnvWrapper],
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
                         `ramos` (summary statistics as proposed in  [1])
        """
        super().__init__(strategy=strategy)

        self._env = remove_all_dr_wrappers(deepcopy(env))
        self._policy = policy

    def __call__(self, dp_values: to.Tensor = None):
        """ Run one rollout, and compute summary statistics. """
        # Don't set the domain params here since they are set by the DomainRandWrapperBuffer to mimic the randomness
        ro = rollout(self._env, self._policy, eval=True)
        ro.torch(data_type=to.get_default_dtype())

        # Return the observations used for inference from the rollout data
        return self.transform_data(ro)
