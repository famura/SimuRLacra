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

from warnings import warn

import torch as to
import torch.nn as nn
from torch.distributions import Normal


class EnergyNoise(nn.Module):
    """Module for learnable additive Gaussian noise with a diagonal covariance matrix"""

    def __init__(
        self,
        noise_dim: [int, tuple],
        energy_net: nn.Module,
        num_steps_chain: int = 100,
        step_size: float = 1e-1,
        noise_std: float = 1e-3,  # TODO IF ETA IS IN THE GAME, THIS HAS TO GO
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param noise_dim: number of dimension
        :param use_cuda: `True` to move the module to the GPU, `False` (default) to use the CPU
        """
        # Call torch.nn.Module's constructor
        super().__init__()

        if not use_cuda:
            self._device = "cpu"
        elif use_cuda and to.cuda.is_available():
            self._device = "cuda"
        elif use_cuda and not to.cuda.is_available():
            warn("Tried to run on CUDA, but it is not available. Falling back to CPU.")
            self._device = "cpu"

        # Register parameters
        self.net = energy_net
        self.prior = EnergyNoise.initial_prior
        self._normal_noise = Normal(to.zeros(noise_dim), noise_std * to.ones(noise_dim))
        self._num_steps_chain = num_steps_chain
        self._step_size = step_size

        # Initialize parameters
        self.reset_expl_params()
        self.to(self.device)

    @staticmethod
    def initial_prior(x: to.Tensor) -> to.Tensor:
        if x.ndim > 1:
            return -to.sum(to.pow(x, 2), dim=1).unsqueeze(1)  # mimic batch processing of nn.Module
        else:
            return -to.sum(to.pow(x, 2)).unsqueeze(0)  # TODO SIGN

    @property
    def device(self) -> str:
        """Get the device (CPU or GPU) on which the policy is stored."""
        return self._device

    def reset_expl_params(self):
        """Reset all parameters of the exploration strategy."""
        self.net.init_param()

    def forward(self, value: to.Tensor) -> to.Tensor:
        """
        Sample from the energy-based distribution.

        :param value: value to evaluate the distribution around
        :return: a detached sample
        """
        # TODO DECIDE WHERE TO START
        # new_value = to.clone(value)
        new_value = to.randn_like(value)

        new_value.requires_grad_(True)

        # Langevin chain
        for i in range(self._num_steps_chain):
            energy_prior = self.prior(new_value)

            energy_likelihood = self.net(new_value)  # == the argument of the exp

            energy = energy_prior + energy_likelihood

            grad = to.autograd.grad(energy.sum(), new_value)[0]

            noise = self._normal_noise.sample()

            new_value = new_value + self._step_size * (grad + noise)  # TODO ETA WOULD ENTER HERE

            # if i % 10 == 0:
            #     print(f"grad norm = {to.norm(grad)}")

        return new_value.detach().requires_grad_(False)

    def log_prob(self, value: to.Tensor) -> to.Tensor:
        r"""
        Calculates the total energy which corresponds to the log-probability.

        :param value: policy parameter values $\theta$ to evaluate at
        :return: the log-probability
        """
        return self.net(value) + self.prior(value)

    def get_entropy(self) -> to.Tensor:
        """
        Get the exploration distribution's entropy.
        The entropy of a normal distribution is independent of the mean.

        :return: entropy value
        """
        raise NotImplementedError
