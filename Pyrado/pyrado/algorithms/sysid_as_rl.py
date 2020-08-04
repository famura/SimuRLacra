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
import itertools
import sys

import numpy as np
import torch as to
import torch.nn as nn
from collections.abc import Iterable
from itertools import product
from typing import Callable, Sequence, Tuple, Dict
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_sampler import ParallelSampler
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult, ParameterSample, \
    ParameterExplorationSampler
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.utils import gen_ordered_batch_idcs
from pyrado.spaces import BoxSpace
from pyrado.spaces.empty import EmptySpace
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import UnitCubeProjector, clamp


class DomainDistrParamPolicy(Policy):
    """ A proxy to the Policy class in order to use the policy's parameters as domain distribution parameters """

    name: str = 'ddp'

    def __init__(self,
                 mapping: Dict[int, Tuple[str, str]],
                 prior: DomainRandomizer = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param mapping: mapping from index of the numpy array (coming from the algorithm) to domain parameter name
                        (e.g. mass, length) and the domain distribution parameter (e.g. mean, std)
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(mapping, dict):
            raise pyrado.TypeErr(given=mapping, expected_type=dict)

        self._mapping = mapping

        # Construct a valid space for the policy parameters aka domain distribution parameters
        bound_lo = -pyrado.inf*np.ones(len(mapping))
        bound_up = +pyrado.inf*np.ones(len(mapping))
        lables = len(mapping)*['None']
        for idx, ddp in self._mapping.items():
            lables[idx] = f'{ddp[0]}_{ddp[1]}'
            if ddp[1] in ['std', 'halfspan']:  # 2nd output is the name of the domain distribution param
                bound_lo[idx] = 1e-6

        param_spec = EnvSpec(obs_space=EmptySpace(), act_space=BoxSpace(bound_lo, bound_up, labels=lables))

        # Call Policy's constructor
        super().__init__(param_spec, use_cuda)

        self.params = nn.Parameter(to.Tensor(param_spec.act_space.flat_dim), requires_grad=True)
        self.init_param(prior=prior)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is not None:
            # First check if there are some specific values to set
            self.param_values = init_values

        elif kwargs.get('prior', None) is not None:
            # Prior information is expected to be in form of a DomainRandomizer since it holds the distributions
            if not isinstance(kwargs['prior'], DomainRandomizer):
                raise pyrado.TypeErr(given=kwargs['prior'], expected_type=DomainRandomizer)

            # For every domain distribution parameter in the mapping, check if there is prior information
            for idx, ddp in self._mapping.items():
                for dp in kwargs['prior'].domain_params:
                    if ddp[0] == dp.name and ddp[1] in dp.get_field_names():
                        # The domain parameter exists in the prior and in the mapping
                        self.params[idx].fill_(getattr(dp, f'{ddp[1]}'))

        else:
            # Last measure
            self.params.data.normal_(0, 1)
            print_cbt('Using uninformative random initialization for DomainDistrParamPolicy.', 'y')

    def clamp_params(self, params: to.Tensor) -> to.Tensor:
        """
        Project the policy parameters a.k.a. the domain distribution parameters to valid a range.

        :param params: parameter tensor with arbitrary values
        :return: parameters clipped to the bounds of the `EnvSpec` defined in the constructor
        """
        return clamp(params,
                     to.from_numpy(self.env_spec.act_space.bound_lo),
                     to.from_numpy(self.env_spec.act_space.bound_up))

    def forward(self, obs: to.Tensor = None) -> to.Tensor:
        # Should never be used. I know this might seem like an extreme abuse of the policy class but it is worth it
        raise NotImplementedError


class SysIdByEpisodicRL(Algorithm):
    """ Wrapper to frame black-box system identification as an episodic reinforcement learning problem """

    name: str = 'rlsysid'
    iteration_key: str = 'rlsysid_iteration'  # logger's iteration key

    def __init__(self,
                 subrtn: Algorithm,
                 behavior_policy: Policy,
                 metric: [Callable[[np.ndarray], np.ndarray], None],
                 obs_dim_weight: [list, np.ndarray],
                 num_rollouts_per_distr: int,
                 num_sampler_envs: int = 4):
        """
        Constructor

        :param subrtn: wrapped algorithm to fit the domain parameter distribution
        :param behavior_policy: lower level policy used to generate the rollouts
        :param metric: from differences in observations to value
        :param num_rollouts_per_distr: number of rollouts per domain distribution parameter set
        :param num_sampler_envs: number of environments for parallel sampling
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
            raise pyrado.ShapeErr(msg=f'Number of policy parameters {subrtn.policy.num_param} does not match the'
                                      f'number of domain distribution parameters {len(subrtn.env.mapping)}!')
        if subrtn.sampler.num_rollouts_per_param != 1:
            # Only sample one rollout in every domain. This is possible since we are synchronizing the init state
            raise pyrado.ValueErr(given=subrtn.sampler.num_rollouts_per_param, eq_constraint='1')
        if num_rollouts_per_distr < 2:
            raise pyrado.ValueErr(given=num_rollouts_per_distr, g_constraint='1')
        if len(obs_dim_weight) != subrtn.env.obs_space.flat_dim:
            raise pyrado.ShapeErr(given=obs_dim_weight, expected_match=subrtn.env.obs_space)

        # Call Algorithm's constructor
        super().__init__(subrtn.save_dir, subrtn.max_iter, subrtn.policy, subrtn.logger)

        # Store inputs
        self._subrtn = subrtn
        self._behavior_policy = behavior_policy
        if metric is not None:
            self.metric = metric
        else:
            w_l1, w_l2 = 0.5, 1.
            self.metric = lambda e: w_l1*np.linalg.norm(e, ord=1, axis=0) + w_l2*np.linalg.norm(e, ord=2, axis=0)
        self.uc_normalizer = UnitCubeProjector(
            bound_lo=subrtn.env.obs_space.bound_lo,
            bound_up=subrtn.env.obs_space.bound_up
        )
        self.obs_dim_weight = np.array(obs_dim_weight)  # weighting factor between the different observations

        # Create the sampler used to execute the same policy as on the real system in the meta-randomized env
        self.behavior_sampler = ParallelSampler(
            self._subrtn.env,
            self._behavior_policy,
            num_envs=num_sampler_envs,
            min_rollouts=1,  # TODO think about this
            seed=1001
        )
        self.num_rollouts_per_distr = num_rollouts_per_distr

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if 'rollouts_real' not in meta_info:
            raise pyrado.KeyErr(key='rollouts_real', container=meta_info)
        if 'init_state' not in meta_info['rollouts_real'][0].rollout_info:  # checking the first element is sufficient
            raise pyrado.KeyErr(key='init_state', container=meta_info['rollouts_real'][0].rollout_info)

        # Extract the initial states from the real rollouts
        rollouts_real = meta_info['rollouts_real']
        init_states_real = [ro.rollout_info['init_state'] for ro in rollouts_real]
        # [ro_r.torch() for ro_r in rollouts_real]

        # Sample new policy parameters a.k.a domain distribution parameters
        param_sets = self._subrtn.expl_strat.sample_param_sets(
            nominal_params=self._subrtn.policy.param_values,
            num_samples=self._subrtn.pop_size,
            include_nominal_params=True
        )

        # Iterate over every domain parameter distribution. We basically mimic the ParameterExplorationSampler here,
        # but we need to adapt the randomizer (and not just the domain parameters) por every policy param set
        param_samples = []
        loss_hist = []
        for idx_ps, ps in enumerate(param_sets):
            # Update the randomizer to use the new
            ps = self._subrtn.policy.clamp_params(ps)
            self._subrtn.env.adapt_randomizer(domain_distr_param_values=ps.detach().numpy())
            self._subrtn.env.randomizer.randomize(num_samples=self.num_rollouts_per_distr)
            sampled_dps = self._subrtn.env.randomizer.get_params()

            # Sample the rollouts
            self.behavior_sampler.set_seed(1001)
            rollouts_sim = self.behavior_sampler.sample(init_states=init_states_real, domain_params=sampled_dps)

            # Iterate over simulated rollout with the same initial state
            for idx_real, idcs_sim in enumerate(gen_ordered_batch_idcs(self.num_rollouts_per_distr, len(rollouts_sim),
                                                                       sorted=True)):
                # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
                ros_real_tr, ros_sim_tr = self.truncate_rollouts([rollouts_real[idx_real]],
                                                                 rollouts_sim[slice(idcs_sim[0], idcs_sim[-1] + 1)])
                assert len(ros_real_tr) == len(ros_sim_tr) == len(idcs_sim)
                assert all([np.allclose(r.rollout_info['init_state'], s.rollout_info['init_state'])
                            for r, s in zip(ros_real_tr, ros_sim_tr)])

                # Concatenate the rollouts to batch the loss computation
                # concat_real = StepSequence.concat(ros_real_tr)
                # concat_sim = StepSequence.concat(ros_sim_tr)
                # loss_per_dim = self.loss_fcn(concat_real, concat_sim)

                for i, (ro_r, ro_s) in enumerate(zip(ros_real_tr, ros_sim_tr)):
                    # Compute the loss for every trajectory
                    loss_per_dim = self.loss_fcn(ro_r, ro_s)
                    loss = self.obs_dim_weight@loss_per_dim
                    loss_hist.append(loss)
                    assert loss >= 0

                    # We need to assign the loss value to the simulated rollout, but this one can be of a different
                    # length than the real-world rollouts as well as of different length than the original
                    # (non-truncated) simulated rollout. We simply put all the loss into the first step (it is an
                    # episodic scenario anyway).
                    rollouts_sim[i + idcs_sim[0]].rewards[:] = 0.
                    rollouts_sim[i + idcs_sim[0]].rewards[0] = -loss

            # Collect the results
            param_samples.append(ParameterSample(params=ps, rollouts=rollouts_sim))

        # Bind the parameter samples and their rollouts in the usual container
        param_samp_res = ParameterSamplingResult(param_samples)

        # Log metrics computed from the old policy (before the update)
        loss_hist = np.asarray(loss_hist)
        self.logger.add_value('min sysid loss', float(np.min(loss_hist)))
        self.logger.add_value('median sysid loss', float(np.median(loss_hist)))
        self.logger.add_value('avg sysid loss', float(np.mean(loss_hist)))
        self.logger.add_value('max sysid loss', float(np.max(loss_hist)))
        self.logger.add_value('std sysid loss', float(np.std(loss_hist)))

        # Extract the best policy parameter sample for saving it later
        self._subrtn.best_policy_param = param_samp_res.parameters[np.argmax(param_samp_res.mean_returns)].clone()

        # Save snapshot data TODO
        # self.make_snapshot(snapshot_mode, float(np.max(param_samp_res.mean_returns)), meta_info)

        # Set the randomizer to domain distribution
        self._subrtn.env.adapt_randomizer(domain_distr_param_values=self._subrtn.best_policy_param.detach().numpy())
        print(self._subrtn.env.randomizer)
        print(self._subrtn.policy.param_values.detach().numpy())

        # Update the wrapped algorithm's update method
        self._subrtn.update(param_samp_res, ret_avg_curr=param_samp_res[0].mean_undiscounted_return)

    def loss_fcn(self, rollout_real: StepSequence, rollout_sim: StepSequence) -> to.Tensor:
        """
        Compute the discrepancy between two time sequences of observations given metric.
        Be sure to align and truncate the rollouts beforehand.

        :param rollout_real: (concatenated) real-world rollout containing the observations
        :param rollout_sim: (concatenated) simulated rollout containing the observations
        :return: (concatenated) discrepancy cost
        """
        if len(rollout_real) != len(rollout_sim):
            raise pyrado.ShapeErr(given=rollout_real, expected_match=rollout_sim)

        # Extract the observations
        real_obs = rollout_real.get_data_values('observations', truncate_last=True)
        sim_obs = rollout_sim.get_data_values('observations', truncate_last=True)

        # Normalize the signals
        real_obs_norm = self.uc_normalizer.project_to(real_obs)
        sim_obs_norm = self.uc_normalizer.project_to(sim_obs)

        # Compute loss based on the error
        return self.metric(real_obs_norm - sim_obs_norm)

    @staticmethod
    def truncate_rollouts(rollouts_real: Sequence[StepSequence],
                          rollouts_sim: Sequence[StepSequence]
                          ) -> Tuple[Sequence[StepSequence], Sequence[StepSequence]]:
        """
        In case (some of the) rollouts failed or succeed in one domain, but not in the other, we truncate the longer
        observation sequence. When truncating, we compare every of the M real rollouts to every of the N simulated
        rollouts, thus replicate the real rollout N times and the simulated rollouts M times.

        :param rollouts_real: M real-world rollouts of different length
        :param rollouts_sim: N simulated rollouts of different length
        :return: MxN real-world rollouts and MxN simulated rollouts of equal length
        """
        if not isinstance(rollouts_real[0], Iterable):
            raise pyrado.TypeErr(given=rollouts_real[0], expected_type=Iterable)
        if not isinstance(rollouts_sim[0], Iterable):
            raise pyrado.TypeErr(given=rollouts_sim[0], expected_type=Iterable)

        # Go over all combinations rollouts individually
        rollouts_real_tr = []
        rollouts_sim_tr = []
        for ro_r, ro_s in product(rollouts_real, rollouts_sim):
            # Handle rollouts of different length, assuming that they are staring at the same state
            if ro_r.length < ro_s.length:
                rollouts_real_tr.append(ro_r)
                rollouts_sim_tr.append(ro_s[:ro_r.length])
            elif ro_r.length > ro_s.length:
                rollouts_real_tr.append(ro_r[:ro_s.length])
                rollouts_sim_tr.append(ro_s)
            else:
                rollouts_real_tr.append(ro_r)
                rollouts_sim_tr.append(ro_s)

        return rollouts_real_tr, rollouts_sim_tr
