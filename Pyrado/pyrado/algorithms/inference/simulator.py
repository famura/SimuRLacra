from itertools import product

import pyrado
import torch as to
from abc import ABC, abstractmethod
from typing import Callable, Union

from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence


class Simulator(Callable, ABC):
    """
    Defines a simulator which maps a parameter-set to an observation
    """

    def __init__(
            self,
            strategy="states"
    ):
        self.strategy = strategy
        self._transformed_representation = False

    def set_representation(self, transformed_representation: bool):
        self._transformed_representation = transformed_representation

    @abstractmethod
    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        raise NotImplementedError

    def transform_data(self, ro: StepSequence, strategy: str = None):
        """
        Transforms rollouts into the desired output.
        Currently a state-representation as well as state-action summary-statistics are available
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
        Calculating summary statistics based on BayesSim's[1] approach. see eq.(22)
        Guarantees output which has the same size for every trajectory.

        [1] Ramos, Fabio, Rafael Carvalhaes Possas, and Dieter Fox.
        "BayesSim: adaptive domain randomization via probabilistic inference for robotics simulators."
        """
        actions = ro.actions.to(dtype=to.float32)
        observations = ro.observations.to(dtype=to.float32)
        # dot product for the state-action dot-product
        obs_diff = observations[1:] - observations[:-1]

        ao_dot_prod = []
        for a, o in product(actions.T, obs_diff.T):
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
        Can be used if each trajectory has the same size or for rnn networks
        """
        return ro.observations.view(-1, 1).squeeze().to(dtype=to.float32)


class EnvSimulator(Simulator):
    """
    Mapping from the environment system parameters to a trajectory-based rollout using a control-policy.
    """
    def __init__(
            self,
            env: Env,
            policy: Policy,
            param_names: list,
            strategy="states"
    ):
        self.name = env.name
        self.env = env
        self.policy = policy
        self.param_names = param_names
        super().__init__(strategy=strategy, )

    def __call__(self, params) -> Union[StepSequence, to.Tensor]:
        ro = rollout(
            self.env,
            self.policy,
            eval=True,
            reset_kwargs=dict(domain_param=dict(zip(self.param_names, params.squeeze().numpy()))),
        )
        ro.torch()
        if self._transformed_representation:
            ro = self.transform_data(ro, strategy=self.strategy)
        # return to.tensor(ro.observations).view(-1, 1).squeeze().to(dtype=to.float32)
        return ro
