from itertools import product

import pyrado
import torch as to
from abc import ABC, abstractmethod
from typing import Union, Mapping

from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence


class RolloutSamplerForSBIBase(ABC):
    """
    Defines a simulator which maps a parameter-set to an observation

    TODO having a base class is nice, but I doubt that we really need it. I mean how many rollout sampler classes should there be. Anyhow, we can leave it for now.
    """

    def __init__(self, strategy: str):
        """
        Constructor

        :param strategy: the method with which the observations are computed from the rollouts. Possible options:
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


class EnvSimulator(RolloutSamplerForSBIBase):
    """
    Mapping from the environment system parameters to a trajectory-based rollout using a control-policy.
    """

    def __init__(self, env: Env, policy: Policy, param_names: list, strategy="states"):
        super().__init__(strategy=strategy)
        self.name = env.name
        self.env = env
        self.policy = policy
        self.param_names = param_names

    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            reset_kwargs=dict(domain_param=dict(zip(self.param_names, params.squeeze().numpy()))),
        )
        ro.torch(data_type=to.get_default_dtype())
        ro = self.transform_data(ro)
        # return to.tensor(ro.observations).view(-1, 1).squeeze()
        return ro


class RolloutSamplerForSBI(RolloutSamplerForSBIBase):
    """
    Mapping from the environment system parameters to a trajectory-based rollout using a control-policy.

    TODO find a better solution. the output should always have the same size.
    TODO this is the place where we choose how to compute the observations from the rollouts, which are then used by sbi
    """

    def __init__(
        self,
        env: Env,
        policy: Policy,
        dp_mapping: Mapping[int, str],
        strategy: str,
    ):
        super().__init__(strategy=strategy)

        self.env = env
        self.policy = policy
        self.param_names = dp_mapping.values()

    def __call__(self, params):
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            reset_kwargs=dict(domain_param=dict(zip(self.param_names, params.squeeze()))),
        )
        ro.torch(data_type=to.get_default_dtype())

        # Return the observations used for inference from the rollout data
        return self.transform_data(ro)


class RealRolloutSamplerForSBI(RolloutSamplerForSBIBase):
    """
    TODO Dirty shit. Bah this is ugly.
    """

    def __init__(
        self,
        env: Env,
        policy: Policy,
        strategy: str,
    ):
        super().__init__(strategy=strategy)

        self.env = env
        self.policy = policy

    def __call__(self, params=None):
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            # Don't set the domain params here since they are set by the DomainRandWrapperBuffer to mimic the randomness
        )
        ro.torch(data_type=to.get_default_dtype())

        # Return the observations used for inference from the rollout data
        return self.transform_data(ro)
