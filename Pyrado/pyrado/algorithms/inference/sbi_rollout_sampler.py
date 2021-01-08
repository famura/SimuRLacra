from itertools import product

import pyrado
import torch as to
from abc import ABC, abstractmethod
from typing import Callable, Union, Mapping

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
                         states, and summary.
        """
        if not strategy.lower() in ["states", "summary"]:
            raise pyrado.ValueErr(given=strategy, eq_constraint="states, summary")

        self.strategy = strategy
        self._transformed_representation = False  # TODO why should we ever not transform?

    def set_representation(self, transformed_representation: bool):
        self._transformed_representation = transformed_representation

    @abstractmethod
    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        raise NotImplementedError

    def transform_data(self, ro: StepSequence, strategy: str = None):
        """
        Transforms rollouts into the observations used for likelihood-free inference.
        Currently a state-representation as well as state-action summary-statistics are available.

        :param ro:
        :param strategy:
        :return:
        """
        strategy = self.strategy if strategy is None else strategy
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
        Computing summary statistics based on approach in [1], see eq. (22).
        This method guarantees output which has the same size for every trajectory.

        [1] Fabio Ramos, Rafael C. Possas, and Dieter Fox. "BayesSim: adaptive domain randomization via probabilistic
            inference for robotics simulators", CONFERENCE?, 2020

        :param ro:
        :return:
        """
        act = ro.actions
        obs = ro.observations
        # dot product for the state-action dot-product
        obs_diff = obs[1:] - obs[:-1]

        ao_dot_prod = []
        for a, o in product(act.T, obs_diff.T):
            ao_dot_prod.append(to.dot(a, o).item())
        ao_dot_prod = to.tensor(ao_dot_prod)

        mean_obs_diff = to.mean(obs_diff, 0)
        var_obs_diff = ((mean_obs_diff - obs_diff) ** 2).mean(dim=0)
        summary_statistics = to.cat((ao_dot_prod, mean_obs_diff, var_obs_diff), 0)
        return summary_statistics

    @staticmethod
    def states_representation(ro: StepSequence) -> to.Tensor:
        """
        Returns the observations of the rollout as one vector.
        Can be used if each trajectory has the same size or for rnn networks.

        :param ro:
        :return:
        """
        return ro.observations.view(-1, 1).squeeze()


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
        ro.torch(data_type=to.float32)
        if self._transformed_representation:
            ro = self.transform_data(ro, strategy=self.strategy)
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
        strategy: str = "states",
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
        ro.torch(data_type=to.float32)

        # Return the observations used for LFI from the rollout data
        return self.transform_data(
            ro[:50], strategy=self.strategy
        )  # TODO only take the first 50 cause the envs are very unlikely to end earlier


class RealRolloutSamplerForSBI(RolloutSamplerForSBIBase):
    """
    TODO Dirty shit. Bah this is ugly.
    """

    def __init__(
        self,
        env: Env,
        policy: Policy,
        strategy: str = "states",
    ):
        super().__init__(strategy=strategy)

        self.env = env
        self.policy = policy

    def __call__(self):
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            # Don't set the domain params here since they are set by the DomainRandWrapperBuffer to mimic the randomness
        )
        ro.torch(data_type=to.float32)

        # Return the observations used for LFI from the rollout data
        return self.transform_data(
            ro[:50], strategy=self.strategy
        )  # TODO only take the first 50 cause the envs are very unlikely to end earlier
