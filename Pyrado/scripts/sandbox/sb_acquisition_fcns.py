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

"""
Plotting Script for evaluating different acquisition functions on a 1-dim toy function
- Probability of Improvement
- Expected Improvement
- Upper Confidence Bound
"""
import torch as to
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyrado import set_seed
from pyrado.utils.functions import noisy_nonlin_fcn
from pyrado.utils.math import UnitCubeProjector


if __name__ == "__main__":
    # Adjustable experiment parameters
    set_seed(1001)
    num_init_samples = 4  # number of initial random points
    num_iter = 6  # number of BO updates
    noise_std = 0.0  # noise level
    acq_fcn = "EI"  # acquisition function (UCB / EI / PI)
    num_acq_restarts = 100  # number of restarts for optimizing the acquisition function
    num_acq_samples = 500  # number of samples for used for optimizing the acquisition function
    ucb_beta = 0.1  # UCB coefficient (only necessary if UCB is used

    # Function boundaries
    x_min_raw, x_max_raw = (-2.0, 5.0)
    x_min, x_max = (0.0, 1.0)
    bounds_raw = to.tensor([[x_min_raw], [x_max_raw]])
    bounds = to.tensor([[x_min], [x_max]])
    uc = UnitCubeProjector(bounds_raw[0, :], bounds_raw[1, :])

    # Generate initial data
    X_train_raw = (x_max_raw - x_min_raw) * to.rand(num_init_samples, 1) + x_min_raw
    y_train_raw = to.from_numpy(noisy_nonlin_fcn(X_train_raw.numpy(), noise_std=noise_std))
    X_train = uc.project_to(X_train_raw)
    y_mean, y_std = y_train_raw.mean(), y_train_raw.std()
    y_train = (y_train_raw - y_mean) / y_std

    # Get best observed value from dataset
    best_observed = [y_train_raw.max().item()]

    # Initialize model
    gp = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # Bayesian Optimization Loop
    for i in tqdm(range(num_iter), total=num_iter):
        print("Iteration:", i + 1)

        # Fit the model
        # mll.train()
        fit_gpytorch_model(mll)
        # mll.eval()
        # gp.eval()

        # Acquisition functions
        ucb = UpperConfidenceBound(gp, beta=ucb_beta, maximize=True)
        ei = ExpectedImprovement(gp, best_f=y_train.max().item(), maximize=True)
        pi = ProbabilityOfImprovement(gp, best_f=y_train.max().item(), maximize=True)
        acq_dict = {"UCB": ucb, "EI": ei, "PI": pi}

        # Optimize acquisition function
        candidate, acq_value = optimize_acqf(
            acq_function=acq_dict[acq_fcn],
            bounds=bounds,
            q=1,
            num_restarts=num_acq_restarts,
            raw_samples=num_acq_samples,
        )
        x_new = candidate.detach()
        x_new_raw = uc.project_back(x_new)

        # Evaluate new candidate
        y_new_raw = noisy_nonlin_fcn(x_new_raw, noise_std=noise_std)

        # Plot the model
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax_gp = fig.add_subplot(gs[0])
        ax_acq = fig.add_subplot(gs[1])
        X_test = to.linspace(x_min, x_max, 501)
        X_test_raw = to.linspace(x_min_raw, x_max_raw, 501)

        with to.no_grad():
            # Get the observations
            y_test_raw = noisy_nonlin_fcn(X_test_raw, noise_std=noise_std)

            # Get the posterior
            posterior = gp.posterior(X_test)
            mean = posterior.mean
            mean_raw = mean * y_test_raw.std() + y_test_raw.mean()
            lower, upper = posterior.mvn.confidence_region()
            lower = lower * y_test_raw.std() + y_test_raw.mean()
            upper = upper * y_test_raw.std() + y_test_raw.mean()

            ax_gp.plot(X_test_raw.numpy(), y_test_raw.numpy(), "k--", label="f(x)")

            ax_gp.plot(X_test_raw.numpy(), mean_raw.numpy(), "b-", lw=2, label="mean")
            ax_gp.fill_between(
                X_test_raw.numpy(), lower.numpy(), upper.numpy(), alpha=0.2, label=r"2$\sigma$ confidence"
            )
            ax_gp.plot(X_train_raw.numpy(), y_train_raw.numpy(), "kx", mew=2, label="samples")

            utility = acq_dict[acq_fcn](X_test[:, None, None])
            ax_acq.plot(X_test_raw.numpy(), utility.numpy(), label=acq_fcn)
            ax_acq.plot(
                X_test_raw.numpy()[utility.argmax().item()],
                utility.max().item(),
                "*",
                markersize=10,
                markerfacecolor="gold",
                markeredgecolor="k",
                markeredgewidth=1,
            )

            if i == num_iter - 1:
                ax_acq.set_xlabel("x")
            if i == 0:
                ax_gp.set_ylabel(r"surrogate $f(x)$")
                ax_acq.set_ylabel(r"utility $\alpha (x)$")
                ax_gp.legend()
                ax_acq.legend()

            fig.suptitle(f"Iteration {i + 1}", y=1.0)
            plt.tight_layout()

        # Update dataset
        X_train_raw = to.cat([X_train_raw, x_new_raw])
        X_train = to.cat([X_train, x_new])  # == uc.project_to(X_train_raw)
        y_train_raw = to.cat([y_train_raw, y_new_raw])
        y_train = (y_train_raw - y_test_raw.mean()) / y_test_raw.std()

        # Update best observed value list
        best_observed.append(y_train_raw.max().item())

        # Reinitialize the models with the updated dataset
        gp = SingleTaskGP(X_train, y_train)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    plt.show()
