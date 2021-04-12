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
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from typing import Sequence, Callable, Optional, Tuple

import pyrado
from pyrado.policies.base import Policy
from pyrado.policies.feed_forward.fnn import FNN
from pyrado.policies.initialization import init_param
from pyrado.utils.data_types import EnvSpec


class MDNPolicy(Policy):
    """
    Conditional Mixture of Densities Network (MDN) using multivariate Gaussian distributions as presented in [1]

    .. seealso::
        [1] G. Papamakarios, I. Murray. "Fast epsilon-free inference of simulation models with Bayesian conditional
            density estimation.", NIPS, 2016
    """

    name: str = "mdn"

    def __init__(
        self,
        spec: EnvSpec,
        num_comp: int,
        hidden_sizes: Sequence[int],
        hidden_nonlin: [Callable, Sequence[Callable]],
        dropout: Optional[float] = 0.0,
        init_param_kwargs: Optional[dict] = None,
        use_cuda: bool = False,
    ):
        """

        :param spec: environment specification
        :param num_comp: number of mixture components
        :param hidden_sizes: sizes of hidden layer outputs. Every entry creates one hidden layer.
        :param hidden_nonlin: nonlinearity for hidden layers
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(num_comp, int) or num_comp < 1:
            raise pyrado.ValueErr(given=num_comp, g_constraint="0 (int)")

        super().__init__(spec, use_cuda)

        # Create the feed-forward neural network
        self.shared = FNN(
            input_size=spec.obs_space.flat_dim,
            output_size=spec.act_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlin=hidden_nonlin,
            dropout=dropout,
            output_nonlin=None,
            use_cuda=use_cuda,
        )

        # Create the different heads
        self.num_comp = num_comp
        self._dim_in = spec.obs_space.flat_dim
        self._dim_out = spec.act_space.flat_dim
        self.coeff_layer = nn.Linear(self._dim_out, self.num_comp)
        self.mean_layer = nn.Linear(self._dim_out, self.num_comp * self._dim_out)
        # A lower triangular (including diagonal) has n * (n+1) / 2 entries
        self.tril_layer = nn.Linear(self._dim_out, self.num_comp * int(self._dim_out * (self._dim_out + 1) / 2))

        # Initialize with no condition
        self._x = None

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)

    def init_param(self, init_values: Optional[to.Tensor] = None, **kwargs):
        if init_values is None:
            # Forward to the FNN's custom initialization function (handles dropout)
            self.shared.init_param(init_values, **kwargs)

            init_param(self.coeff_layer, **kwargs)
            init_param(self.tril_layer, **kwargs)
            init_param(self.mean_layer, **kwargs)

        else:
            self.param_values = init_values

    def forward(self, x: to.Tensor) -> Tuple[to.Tensor, to.Tensor, to.Tensor]:
        x = x.to(device=self.device, dtype=to.get_default_dtype())

        # Get latent from the FNN
        z = self.shared(x)

        # Compute the components' mixture coefficients
        coeffs = nn.functional.softmax(self.coeff_layer(z), -1)

        # Compute the components' means
        means = self.mean_layer(z).reshape(-1, self._dim_out, self.num_comp)

        # Compute the lower triangular matrices' elements of shape [batch_size, num_tril_ele, num_comp]
        trils = self.tril_layer(z).reshape(-1, int(self._dim_out * (self._dim_out + 1) / 2), self.num_comp)

        return coeffs, means, trils

    def log_prob(self, y: to.Tensor, x: Optional[to.Tensor] = None, **kwargs):
        """
        Calculates the log-probability of the mixture of experts.
        The posterior is normalized TODO: check this

        :param y: sample y for which the log-probability should be calculated
        :param x: a condition
        :return: TODO
        """
        if y.ndim == 1:
            y = y.unsqueeze(dim=0)
        if x is None or not y.shape[0] == x.shape[0]:
            x = self._check_single_x(x)

        pi, mu, tril = self.forward(x)
        scale_tril = self._compose_cov_from_tril(tril)
        log_probs = to.zeros((y.shape[0], self.num_comp))
        log_probs_sum = to.zeros((y.shape[0],))

        for idx in range(self.num_comp):
            mvgaussian = MultivariateNormal(loc=mu[:, :, idx], scale_tril=scale_tril[:, :, :, idx])
            log_prob = mvgaussian.log_prob(y)
            assert log_prob.shape == (y.shape[0],)
            log_probs[:, idx] = log_prob + pi.T[idx].log()
            log_probs_sum += log_prob + pi.T[idx].log()

        return to.logsumexp(log_probs, dim=1)

    def sample(self, sample_shape: Tuple, x: Optional[to.Tensor] = None, **kwargs):
        x = self._check_single_x(x)

        self.eval()
        pi, mu, tril = self.forward(x)
        scale_tril = self._compose_cov_from_tril(tril)
        mog = [
            MultivariateNormal(loc=m, scale_tril=s) for m, s in zip(mu.permute(2, 0, 1), scale_tril.permute(3, 0, 1, 2))
        ]

        # Categorical of mixing coefficients
        categorical = Categorical(probs=pi)

        # Sample from categorical to choose which gaussian should be used
        sample_which_normal = categorical.sample(sample_shape)

        # Sample from the Gaussians
        samples = [mog[i].sample().squeeze() for i in sample_which_normal]
        samples = to.stack(samples)
        self.train()

        return samples

    def _compose_cov_from_tril(self, tril: to.Tensor) -> to.Tensor:
        r"""
        Constructs covariance matrixes from the the cholesky decomposition of the covariance matrix
        $\Sigma = G G^T = L D L^T$, where G := L D^{1/2},
        with L being the normalized and G being the un-normalized lower triangular matrices.

        .. note::
            `to.einsum("ij,kj->ik", A, B)` equals `to.matmul(A, B.T)`

        :param tril: lower triangular elements (including diagonal) of the Cholesky factorization
        :return: composed covariance matrix as a 4-dim tensor of shape [batch_size, n, n, num_comp]
        """
        # Bring the lower triangular into the shape of the covariance
        idcs_tril = np.tril_indices(self._dim_out)
        cov_tril = to.zeros(tril.shape[0], self._dim_out, self._dim_out, self.num_comp)
        cov_tril[:, idcs_tril[0], idcs_tril[1], :] = tril

        # Compute L L^T for the inner two dimensions
        cov = to.einsum("aijb,akjb->aikb", cov_tril, cov_tril)

        # Check for the 1st of the remaining dims
        tmp = cov_tril[0, :, :, 0].clone()
        assert to.all(to.isclose(cov[0, :, :, 0], to.matmul(tmp, tmp.T)))

        # Return the full covariance matrix
        return cov

    def _check_single_x(self, x: Optional[to.Tensor]):
        r"""
        Returns the correct shape of the condition $x$ in case a single $x$ is required.
        If `x` is `None`, the default condition is returned.

        :param x: condition
        :return: checked condition or default condition
        """
        if x is None:
            if self._x is not None:
                return self._x
            else:
                raise pyrado.ValueErr(
                    given=x, msg="No condition x was given, neither was it set to default in advance."
                )
        elif x.ndim == 1:
            return x.unsqueeze(dim=0)
        elif x.ndim == 2 and x.shape == (1, self._dim_in):
            return x
        else:
            raise pyrado.ShapeErr(given=x, msg=f"Expected shape (1, {self._dim_in}), but received {x.shape}")

    def set_default_x(self, x: to.Tensor):
        """
        Set a default posterior condition. This function mimics the one from the sbi package.

        :param x: condition
        """
        self._x = x
