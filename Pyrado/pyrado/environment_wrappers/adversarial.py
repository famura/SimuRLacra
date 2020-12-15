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

from abc import ABC
import numpy as np
import torch as to

from init_args_serializer import Serializable
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environment_wrappers.utils import inner_env, typed_env


class AdversarialWrapper(EnvWrapper, ABC):
    """ Base class for adversarial wrappers (used in ARPL) """

    def __init__(self, wrapped_env, policy, eps, phi):
        EnvWrapper.__init__(self, wrapped_env)
        self._policy = policy
        self._eps = eps
        self._phi = phi

    @staticmethod
    def quadratic_loss(action):
        return to.norm(action).pow(2)

    def decide_apply(self):
        return np.random.binomial(1, self._phi) == 1

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, val):
        self._phi = val


class AdversarialObservationWrapper(AdversarialWrapper, Serializable):
    """" Wrapper to apply adversarial perturbations to the observations (used in ARPL) """

    def __init__(self, wrapped_env, policy, eps, phi):
        """
        Constructor

        :param wrapped_env: environment to be wrapped
        :param policy: policy to be updated
        :param eps: magnitude of perturbation
        :param phi: probability of perturbation
        """
        Serializable._init(self, locals())
        AdversarialWrapper.__init__(self, wrapped_env, policy, eps, phi)

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        adversarial = self.get_arpl_grad(obs)
        if self.decide_apply():
            obs += adversarial.view(-1).float().numpy()
        return obs, reward, done, info

    def get_arpl_grad(self, state):
        state_tensor = to.tensor([state], requires_grad=True, dtype=to.double)
        mean_arpl = self._policy.forward(state_tensor)
        l2_norm_mean = -to.norm(mean_arpl, p=2, dim=1)
        l2_norm_mean.backward()
        state_grad = state_tensor.grad
        return self._eps * to.sign(state_grad)


class AdversarialStateWrapper(AdversarialWrapper, Serializable):
    """" Wrapper to apply adversarial perturbations to the state (used in ARPL) """

    def __init__(self, wrapped_env, policy, eps, phi, torch_observation=False):
        """
        Constructor

        :param wrapped_env: environment to be wrapped
        :param policy: policy to be updated
        :param eps: magnitude of perturbation
        :param phi: probability of perturbation
        :param torch_observation: observation uses torch
        """
        Serializable._init(self, locals())
        AdversarialWrapper.__init__(self, wrapped_env, policy, eps, phi)
        self.torch_observation = torch_observation

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        saw = typed_env(self.wrapped_env, StateAugmentationWrapper)
        state = inner_env(self).state
        nonobserved = to.from_numpy(obs[saw.offset :])
        adversarial = self.get_arpl_grad(state, nonobserved)
        if self.decide_apply():
            inner_env(self).state += adversarial.view(-1).numpy()
        if saw:
            obs[: saw.offset] = inner_env(self).observe(inner_env(self).state)
        else:
            obs = inner_env(self).observe(inner_env(self).state)
        return obs, reward, done, info

    def get_arpl_grad(self, state, nonobserved):
        if isinstance(state, np.ndarray):
            state_tensor = to.tensor(state, requires_grad=True)
        elif isinstance(state, to.Tensor):
            state_tensor = state
        else:
            raise ValueError("state could not be converted to a torch tensor")
        if self.torch_observation:
            observation = inner_env(self).observe(state_tensor, dtype=to.Tensor)
        else:
            observation = state_tensor
        mean_arpl = self._policy.forward(to.cat((observation, nonobserved)))
        l2_norm_mean = -to.norm(mean_arpl, p=2, dim=0)
        l2_norm_mean.backward()
        state_grad = state_tensor.grad
        return self._eps * to.sign(state_grad)


class AdversarialDynamicsWrapper(AdversarialWrapper, Serializable):
    """" Wrapper to apply adversarial perturbations to the domain parameters (used in ARPL) """

    def __init__(self, wrapped_env, policy, eps, phi, width=0.25):
        """
        Constructor

        :param wrapped_env: environemnt to be wrapped
        :param policy: policy to be updated
        :param eps: magnitude of perturbation
        :param phi: probability of perturbation
        :param width: width of distribution to sample from
        """
        Serializable._init(self, locals())
        AdversarialWrapper.__init__(self, wrapped_env, policy, eps, phi)
        self.width = width
        self.saw = typed_env(self.wrapped_env, StateAugmentationWrapper)
        self.nominal = self.saw.nominal
        self.nominalT = to.from_numpy(self.nominal)
        self.adv = None
        self.re_adv()

    def re_adv(self):
        self.adv = np.random.uniform(1 - self.width, 1 + self.width, self.nominal.shape) * self.nominal

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        self.re_adv()
        self.saw.set_param(to.tensor(self.adv))
        return self.wrapped_env.reset(init_state, domain_param)

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        state = obs.clone()
        adversarial = self.get_arpl_grad(state) * self.nominalT
        if self.decide_apply():
            new_params = to.tensor(self.adv).squeeze(0) + adversarial
            self.saw.set_param(new_params.squeeze(0))
        return obs, reward, done, info

    def get_arpl_grad(self, state):
        state_tensor = to.tensor([state], requires_grad=True)
        self.saw.set_param(self.adv)
        mean_arpl = self._policy.forward(state_tensor)
        l2_norm_mean = -to.norm(mean_arpl, p=2, dim=1)
        l2_norm_mean.backward()
        state_grad = state_tensor.grad
        state_grad = state_grad[:, self.saw.offset :]
        return self._eps * to.sign(state_grad)
