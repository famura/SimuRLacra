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
from typing import Union

import torch as to
import torch.nn as nn
from abc import ABC
from torch.distributions import Distribution, Bernoulli, Categorical

import pyrado
from pyrado.policies.base import Policy
from pyrado.exploration.normal_noise import DiagNormalNoise
from pyrado.exploration.uniform_noise import UniformNoise
from pyrado.policies.two_headed import TwoHeadedPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.math import clamp
from pyrado.utils.properties import Delegate
from pyrado.utils.tensor import atleast_2D


class StochasticActionExplStrat(Policy, ABC):
    """ Explore by sampling actions from a distribution. """

    def __init__(self, policy: Policy):
        """
        Constructor

        :param policy: wrapped policy
        """
        super().__init__(policy.env_spec, use_cuda=policy.device == 'cuda')
        self.policy = policy

    @property
    def is_recurrent(self) -> bool:
        return self.policy.is_recurrent

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        return self.policy.init_hidden(batch_size)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        self.policy.init_param(init_values, **kwargs)

    def reset(self):
        self.policy.reset()

    def forward(self, obs: to.Tensor, *extra) -> (to.Tensor, tuple):
        # Get actions from policy
        if self.policy.is_recurrent:
            if isinstance(self.policy, TwoHeadedPolicy):
                act, other, hidden = self.policy(obs, *extra)
            else:
                act, hidden = self.policy(obs, *extra)
        else:
            if isinstance(self.policy, TwoHeadedPolicy):
                act, other = self.policy(obs, *extra)
            else:
                act = self.policy(obs, *extra)

        # Compute exploration (use rsample to apply the reparametrization trick  if needed)
        act_expl = self.action_dist_at(act).rsample()  # act is the mean if train_mean=False

        # Return the exploratove actions and optionally the other policy outputs
        if self.policy.is_recurrent:
            if isinstance(self.policy, TwoHeadedPolicy):
                return act_expl, other, hidden
            else:
                return act_expl, hidden
        else:
            if isinstance(self.policy, TwoHeadedPolicy):
                return act_expl, other
            else:
                return act_expl

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> Distribution:
        """
        Re-evaluate the given rollout using the policy wrapped by this exploration strategy.
        Use this method to get gradient data on the action distribution.

        :param rollout: complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks
        :return: actions with gradient data
        """
        self.policy.eval()
        if isinstance(self.policy, TwoHeadedPolicy):
            acts, _ = self.policy.evaluate(rollout, hidden_states_name)  # ignore the second head's output
        else:
            acts = self.policy.evaluate(rollout, hidden_states_name)
        return self.action_dist_at(acts)

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        """
        Return the action distribution for the given output from the wrapped policy.

        :param policy_output: output from the wrapped policy, i.e. the noise-free action values
        :return: action distribution
        """
        raise NotImplementedError


class NormalActNoiseExplStrat(StochasticActionExplStrat):
    """ Exploration strategy which adds Gaussian noise to the continuous policy actions """

    def __init__(self,
                 policy: Policy,
                 std_init: [float, to.Tensor],
                 std_min: [float, to.Tensor] = 1e-3,
                 train_mean: bool = False,
                 learnable: bool = True):
        """
        Constructor

        :param policy: wrapped policy
        :param std_init: initial standard deviation for the exploration noise
        :param std_min: minimal standard deviation for the exploration noise
        :param train_mean: set `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        """
        super().__init__(policy)

        self._noise = DiagNormalNoise(
            use_cuda=policy.device == 'cuda',
            noise_dim=policy.env_spec.act_space.flat_dim,
            std_init=std_init,
            std_min=std_min,
            train_mean=train_mean,
            learnable=learnable
        )

    @property
    def noise(self) -> DiagNormalNoise:
        """ Get the exploration noise. """
        return self._noise

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        return self._noise(policy_output)

    # Make NormalActNoiseExplStrat appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    std = Delegate('_noise')
    mean = Delegate('_noise')
    get_entropy = Delegate('_noise')


class UniformActNoiseExplStrat(StochasticActionExplStrat):
    """ Exploration strategy which adds uniform noise to the continuous policy actions """

    def __init__(self,
                 policy: Policy,
                 halfspan_init: [float, to.Tensor],
                 halfspan_min: [float, list] = 0.01,
                 train_mean: bool = False,
                 learnable: bool = True):
        """
        Constructor

        :param policy: wrapped policy
        :param halfspan_init: initial value of the half interval for the exploration noise
        :param halfspan_min: minimal standard deviation for the exploration noise
        :param train_mean: set `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        """
        super().__init__(policy)

        self._noise = UniformNoise(
            use_cuda=policy.device == 'cuda',
            noise_dim=policy.env_spec.act_space.flat_dim,
            halfspan_init=halfspan_init,
            halfspan_min=halfspan_min,
            train_mean=train_mean,
            learnable=learnable
        )

    @property
    def noise(self) -> UniformNoise:
        """ Get the exploration noise. """
        return self._noise

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        return self._noise(policy_output)

    # Make NormalActNoiseExplStrat appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    halfspan = Delegate('_noise')
    get_entropy = Delegate('_noise')


class SACExplStrat(StochasticActionExplStrat):
    """
    State-dependent exploration strategy which adds normal noise squashed into by a tanh to the continuous actions.

    .. note::
        This exploration strategy is specifically designed for SAC.
        Due to the tanh transformation, it returns action values within [-1,1].
    """

    def __init__(self, policy: Policy):
        """
        Constructor

        :param policy: wrapped policy
        """
        if not isinstance(policy, TwoHeadedPolicy):
            raise pyrado.TypeErr(given=policy, expected_type=TwoHeadedPolicy)

        super().__init__(policy)

        # Do not need to learn the exploration noise via an optimizer, since it is handled by the policy in this case
        self._noise = DiagNormalNoise(
            use_cuda=policy.device == 'cuda',
            noise_dim=policy.env_spec.act_space.flat_dim,
            std_init=1.,  # std_init will be overwritten by 2nd policy head
            std_min=0.,  # ignore since we are explicitly clipping in log space later
            train_mean=False,
            learnable=False
        )

        self._log_std_min = to.tensor(-20.)  # approx 2.061e-10
        self._log_std_max = to.tensor(2.)  # approx 7.389

    @property
    def noise(self) -> DiagNormalNoise:
        """ Get the exploration noise. """
        return self._noise

    def action_dist_at(self, policy_out_1: to.Tensor, policy_out_2: to.Tensor) -> Distribution:
        """
        Return the action distribution for the given output from the wrapped policy.
        This method is made for two-headed policies, e.g. used with SAC.

        :param policy_out_1: first head's output from the wrapped policy, noise-free action values
        :param policy_out_2: second head's output from the wrapped policy, state-dependent log std values
        :return: action distribution at the mean given by `policy_out_1`
        """
        # Manually adapt the Gaussian's variance to the clipped value
        log_std = clamp(policy_out_2, lo=self._log_std_min, up=self._log_std_max)
        self._noise.std = to.exp(log_std)

        return self._noise(policy_out_1)

    # Make NormalActNoiseExplStrat appear as if it would have the following functions / properties
    reset_expl_params = Delegate('_noise')
    std = Delegate('_noise')
    mean = Delegate('_noise')
    get_entropy = Delegate('_noise')

    def forward(self, obs: to.Tensor, *extra) -> [(to.Tensor, to.Tensor), (to.Tensor, to.Tensor, to.Tensor)]:
        # Get actions from policy (which for this class always have a two-headed architecture)
        if self.policy.is_recurrent:
            act, log_std, hidden = self.policy(obs, *extra)
        else:
            act, log_std = self.policy(obs, *extra)

        # Compute exploration (use rsample to apply the reparametrization trick)
        act_noise_distr = self.action_dist_at(act, log_std)
        u = act_noise_distr.rsample()
        act_expl = to.tanh(u)  # is in [-1, 1], this is why we always use an ActNormWrapper for SAC
        log_prob = act_noise_distr.log_prob(u)
        log_prob = self._enforce_act_expl_bounds(log_prob, act_expl)

        # Return the action and the log of the exploration std, given the current observation
        if self.policy.is_recurrent:
            return act_expl, log_prob, hidden
        else:
            return act_expl, log_prob

    @staticmethod
    def _enforce_act_expl_bounds(log_probs: to.Tensor, act_expl: to.Tensor, eps: float = 1e-6):
        r"""
        Transform the `log_probs` accounting for the squashed tanh exploration.

        .. seealso::
            Eq. (21) in [2]

        :param log_probs: $\log( \mu(u|s) )$
        :param act_expl: action values with explorative noise
        :param eps: additive term for numerical stability of the logarithm function
        :return: $\log( \pi(a|s) )$
        """
        # Batch dim along the first dim
        act_expl_ = atleast_2D(act_expl)
        log_probs_ = atleast_2D(log_probs)

        # Sum over action dimensions
        log_probs_ = to.sum(log_probs_ - to.log(to.ones_like(act_expl_) - to.pow(act_expl_, 2) + eps), 1, keepdim=True)
        if act_expl_.shape[0] > 1:
            return log_probs_  # batched mode
        else:
            return log_probs_.squeeze(1)  # one sample at a time

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> Distribution:
        """
        Re-evaluate the given rollout using the policy wrapped by this exploration strategy.
        Use this method to get gradient data on the action distribution.
        This version is tailored to the two-headed policy architecture used for SAC, since it requires a two-headed
        policy, where the first head returns the mean action and the second head returns the state-dependent std.

        :param rollout: complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks
        :return: actions with gradient data
        """
        self.policy.eval()
        acts, log_stds = self.policy.evaluate(rollout, hidden_states_name)
        return self.action_dist_at(acts, log_stds)


class EpsGreedyExplStrat(StochasticActionExplStrat):
    """ Exploration strategy which selects discrete actions epsilon-greedily """

    def __init__(self, policy: Policy, eps: float = 1., eps_schedule_gamma: float = 0.99, eps_final: float = 0.05):
        """
        Constructor

        :param policy: wrapped policy
        :param eps: parameter determining the greediness, can be optimized or scheduled
        :param eps_schedule_gamma: temporal discount factor for the exponential decay of epsilon
        :param eps_final: minimum value of epsilon
        """
        super().__init__(policy)

        self.eps = nn.Parameter(to.tensor(eps), requires_grad=True)
        self._eps_init = to.tensor(eps)
        self._eps_final = to.tensor(eps_final)
        self._eps_old = to.tensor(eps)
        self.eps_gamma = eps_schedule_gamma
        self.distr_eps = Bernoulli(probs=self.eps.data)  # eps chance to sample 1

        flat_dim = self.policy.env_spec.act_space.flat_dim
        self.distr_act = Categorical(to.ones(flat_dim)/flat_dim)

    def eval(self):
        """ Call PyTorch's eval function and set the deny every exploration. """
        super(Policy, self).eval()
        self._eps_old = self.eps.clone()
        self.eps.data = to.tensor(0.)
        self.distr_eps = Bernoulli(probs=self.eps.data)

    def train(self, mode=True):
        """ Call PyTorch's eval function and set the re-activate every exploration. """
        super(Policy, self).train()
        self.eps = nn.Parameter(self._eps_old, requires_grad=True)
        self.distr_eps = Bernoulli(probs=self.eps.data)

    def schedule_eps(self, steps: int):
        self.eps.data = self._eps_final + (self._eps_init - self._eps_final)*self.eps_gamma**steps
        self.distr_eps = Bernoulli(probs=self.eps.data)

    def forward(self, obs: to.Tensor, *extra) -> (to.Tensor, tuple):
        # Get exploiting action from policy given the current observation (this way we always get a value for hidden)
        if self.policy.is_recurrent:
            act, hidden = self.policy(obs, *extra)
        else:
            act = self.policy(obs, *extra)

        # Compute epsilon-greedy exploration
        if self.distr_eps.sample() == 1:
            act_idx = self.distr_act.sample()
            act = self.env_spec.act_space.eles[int(act_idx)]
            act = to.from_numpy(act).to(to.get_default_dtype())

        if self.policy.is_recurrent:
            return act, hidden
        else:
            return act

    def action_dist_at(self, policy_output: to.Tensor) -> Distribution:
        # Not needed for this exploration strategy
        raise NotImplementedError
