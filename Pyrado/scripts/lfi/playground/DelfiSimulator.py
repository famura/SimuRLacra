import numpy as np
from typing import Optional, Callable, Type, Mapping, Tuple
import torch as to

from delfi.simulator.BaseSimulator import BaseSimulator

from pyrado.policies.base import Policy
from pyrado.environments.sim_base import SimEnv
from pyrado.algorithms.inference.sbi_rollout_sampler import SimRolloutSamplerForSBI


class DelfiSimulator(BaseSimulator):
    def __init__(
        self,
        sim_env: SimEnv,
        policy: Policy,
        dp_mapping: Mapping[int, str],
        summary_statistic: str,
        seed=None,
        init_states_real=None):

        self._env_sim = sim_env
        self._policy = policy
        self.dp_mapping = dp_mapping
        self.summary_statistic = summary_statistic.lower()

        self.rollout_sampler = self.rollout_sampler = SimRolloutSamplerForSBI(
            self._env_sim, self._policy, self.dp_mapping, self.summary_statistic, init_states_real)

        dim_param = len(self._env_sim.domain_param)
        super().__init__(dim_param=dim_param, seed=seed)

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = np.asarray(params)

        assert params.ndim == 1, 'params.ndim must be 1'

        states = self.rollout_sampler(to.Tensor(params))

        return {'data': states.reshape(-1),
                'dt': self._env_sim.dt}
