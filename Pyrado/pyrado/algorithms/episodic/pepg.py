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
from typing import Optional

from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.utils.math import clamp_symm
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult
from pyrado.exploration.stochastic_params import SymmParamExplStrat, NormalParamNoise


def rank_transform(arr: np.ndarray, centered=True) -> np.ndarray:
    """
    Transform a 1-dim ndarray with arbitrary scalar values to an array with equally spaced rank values.
    This is a nonlinear transform.

    :param arr: input array
    :param centered: if the transform should by centered around zero
    :return: transformed array
    """
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1

    # Create array to sort in
    ranks = np.zeros_like(arr)
    # Ascending sort
    idcs_sort = np.argsort(arr)
    # Rearrange to an equal-step array from -0.5 (or 0) to 0.5 (or 1)
    if centered:
        ranks[idcs_sort] = np.linspace(-.5, .5, idcs_sort.size, endpoint=True)
    else:
        ranks[idcs_sort] = np.linspace(0., 1., idcs_sort.size, endpoint=True)
    return ranks


class PEPG(ParameterExploring):
    """
    Parameter-Exploring Policy Gradients (PEPG)

    .. seealso::
        [1] F. Sehnke, C. Osendorfer, T. Rueckstiess, A. Graves, J. Peters, J. Schmidhuber, "Parameter-exploring
        Policy Gradients", Neural Networks, 2010
    """

    name: str = 'pepg'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 num_rollouts: int,
                 expl_std_init: float,
                 expl_std_min: float = 0.01,
                 pop_size: Optional[int] = None,
                 clip_ratio_std: float = 0.05,
                 normalize_update: bool = False,
                 transform_returns: bool = True,
                 lr: float = 5e-4,
                 num_workers: int = 4,
                 logger: Optional[StepLogger] = None):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param pop_size: number of solutions in the population
        :param num_rollouts: number of rollouts per policy sample
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param clip_ratio_std: maximal ratio for the change of the exploration strategy's standard deviation
        :param transform_returns: use a rank-transformation of the returns to update the policy
        :param lr: learning rate
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call ParameterExploring's constructor
        super().__init__(
            save_dir,
            env,
            policy,
            max_iter,
            num_rollouts,
            pop_size=pop_size,
            num_workers=num_workers,
            logger=logger
        )

        # Store the inputs
        self.clip_ratio_std = clip_ratio_std
        self.normalize_update = normalize_update
        self.transform_returns = transform_returns
        self.lr = lr

        # Exploration strategy based on symmetrical normally distributed noise
        if self.pop_size%2 != 0:
            # Symmetric buffer needs to have an even number of samples
            self.pop_size += 1
        self._expl_strat = SymmParamExplStrat(NormalParamNoise(
            self._policy.num_param,
            std_init=expl_std_init,
            std_min=expl_std_min,
        ))

        self.optim = to.optim.SGD([{'params': self._policy.parameters()}], lr=lr, momentum=0.8, dampening=0.1)

    @to.no_grad()
    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = param_results[1:].mean_returns

        # Rank policy parameters by return (a.k.a. fitness)
        rets = rank_transform(rets_avg_ros) if self.transform_returns else rets_avg_ros

        # Move to PyTorch
        rets = to.from_numpy(rets).to(to.get_default_dtype())
        rets_max = to.max(rets)
        rets_avg_symm = (rets[:len(param_results)//2] + rets[len(param_results)//2:])/2.
        baseline = to.mean(rets)  # zero if centered

        # Compute finite differences for the average return of each solution
        rets_fds = rets[:len(param_results)//2] - rets[len(param_results)//2:]

        # Get the perturbations (select the first half since they are symmetric)
        epsilon = param_results.parameters[:len(param_results)//2, :] - self._policy.param_values

        if self.normalize_update:
            # See equation (15, top) in [1]
            delta_mean = (rets_fds/(2*rets_max - rets_fds + 1e-6))@epsilon  # epsilon = T from [1]
        else:
            # See equation (13) in [1]
            delta_mean = 0.5*rets_fds@epsilon  # epsilon = T from [1]

        # Update the mean
        self.optim.zero_grad()
        self._policy.param_grad = -delta_mean  # PyTorch optimizers are minimizers
        self.optim.step()
        # Old version without PyTorch optimizer: self._expl_strat.policy.param_values += delta_mean * self.lr

        # Update the std
        S = (epsilon**2 - self._expl_strat.std**2)/self._expl_strat.std

        if self.normalize_update:
            # See equation (15, bottom) in [1]
            delta_std = (rets_avg_symm - baseline)@S
        else:
            # See equation (14) in [1]
            delta_std = ((rets_avg_symm - baseline)/(rets_max - baseline + 1e-6))@S

        # Bound the change on the exploration standard deviation (i.e. the entropy)
        delta_std *= self.lr
        delta_std = clamp_symm(delta_std, self.clip_ratio_std*self._expl_strat.std)
        new_std = self._expl_strat.std + delta_std

        self._expl_strat.adapt(std=new_std)

        # Logging
        self.logger.add_value('policy param', self._policy.param_values, 4)
        self.logger.add_value('delta policy param', delta_mean*self.lr, 4)
        self.logger.add_value('expl strat std', self._expl_strat.std, 4)
        self.logger.add_value('expl strat entropy', self._expl_strat.get_entropy(), 4)
