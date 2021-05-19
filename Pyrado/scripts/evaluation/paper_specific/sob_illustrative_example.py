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
This script yields the values for the illustrative example in

.. seealso::
    [1] F. Muratore, M. Gienger, J. Peters, "Assessing Transferability from Simulation to Reality for Reinforcement
        Learning", PAMI, 2021
"""
import os
import os.path as osp

import numpy as np
from matplotlib import pyplot as plt
from scipy import special

import pyrado
from pyrado import set_seed
from pyrado.environments.one_step.catapult import CatapultExample
from pyrado.plotting.curve import draw_curve_from_data
from pyrado.utils.argparser import get_argparser


def calc_E_n_Jhat(n, th):
    r"""
    Calculate $E_\\xi[ \hat{J}_n(\theta) ]$ approximated by $sum_{i=1}^n p(\\xi_i) \hat{J}_n(\theta)$.

    :param n: number of domains $n$ to approximate the expectation
    :param th: (arbitrary) policy parameter, might be estimated using n domain parameters, but does not have to be
    :return: approximation of $E_\\xi[ \hat{J}_n(\theta) ]$
    """
    E_n_Jhat_th = 0
    for i in range(n + 1):
        # i is the number of Venus draws
        binom_coeff = special.binom(n, i)
        E_n_Jhat_th += binom_coeff * pow(psi, i) * pow(1 - psi, n - i) * env.est_expec_return(th, n - i, i)
    return E_n_Jhat_th


def calc_E_n_Jhat_th_opt(n):
    r"""
    Calculate $E_\\xi[ \hat{J}_n(\theta^*) ]$ approximated by $sum_{i=1}^n p(\\xi_i) \hat{J}_n(\theta^*)$.

    :param n: number of domains $n$ to approximate the expectation
    :return: approximation of $E_\\xi[ \hat{J}_n(\theta^*) ]$
    """
    E_n_Jhat_th_opt = 0
    for i in range(n + 1):
        # i is the number of Venus draws
        binom_coeff = special.binom(n, i)
        E_n_Jhat_th_opt += binom_coeff * pow(psi, i) * pow(1 - psi, n - i) * env.opt_est_expec_return(n - i, i)
    return E_n_Jhat_th_opt


def check_E_n_Jhat(th_n_opt, n):
    """
    Check the influence of the number of domains $n$ used for the expectation operator.

    :param th_n_opt: optimal policy parameter determined from n domains
    :param n: number of domains $n$ used for determining the policy parameters
    """
    # "Manual" expectation using n=3 domain parameters
    E_3_Jhat_n_opt = (
        1 * pow(psi, 3) * env.est_expec_return(th_n_opt, 0, 3)
        + 3 * pow(psi, 2) * pow(1 - psi, 1) * env.est_expec_return(th_n_opt, 1, 2)
        + 3 * pow(psi, 1) * pow(1 - psi, 2) * env.est_expec_return(th_n_opt, 2, 1)
        + 1 * pow(1 - psi, 3) * env.est_expec_return(th_n_opt, 3, 0)
    )
    print(f"E_3_Jhat_{n}_opt:   {E_3_Jhat_n_opt}")

    # Expectation using n=50 domain parameters
    E_3_Jhat_n_opt = calc_E_n_Jhat(3, th_n_opt)
    print(f"E_3_Jhat_{n}_opt:   {E_3_Jhat_n_opt}")

    # Expectation using n=50 domain parameters
    E_50_Jhat_n_opt = calc_E_n_Jhat(50, th_n_opt)
    print(f"E_50_Jhat_{n}_opt:  {E_50_Jhat_n_opt}")

    # Expectation using n=500 domain parameters
    E_500_Jhat_n_opt = calc_E_n_Jhat(500, th_n_opt)
    print(f"E_500_Jhat_{n}_opt: {E_500_Jhat_n_opt}")


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Set up the example
    ex_dir = osp.join(pyrado.EVAL_DIR, "illustrative_example")
    env = CatapultExample(m=1.0, g_M=3.71, k_M=1000.0, x_M=0.5, g_V=8.87, k_V=3000.0, x_V=1.5)
    psi = 0.7  # true probability of drawing Venus
    num_samples = 100
    num_iter = 30
    noise_th_scale = 0.15
    set_seed(args.seed)
    fig_size = tuple([0.75 * x for x in pyrado.figsize_thesis_1percol_18to10])

    th_true_opt = env.opt_policy_param(1 - psi, psi)  # true probabilities instead of counts
    J_true_opt = env.opt_est_expec_return(1 - psi, psi)  # true probabilities instead of counts
    print(f"th_true_opt: {th_true_opt}")
    print(f"J_true_opt:  {J_true_opt}\n")

    # Initialize containers
    n_M_hist = np.empty((num_samples, num_iter))
    n_V_hist = np.empty((num_samples, num_iter))
    th_n_opt_hist = np.empty((num_samples, num_iter))
    th_c_hist = np.empty((num_samples, num_iter))
    Jhat_th_n_opt_hist = np.empty((num_samples, num_iter))
    Jhat_th_c_hist = np.empty((num_samples, num_iter))
    Jhat_th_true_opt_hist = np.empty((num_samples, num_iter))
    G_n_hist = np.empty((num_samples, num_iter))
    G_true_hist = np.empty((num_samples, num_iter))
    b_Jhat_n_hist = np.empty((num_samples, num_iter))

    for s in range(num_samples):

        for n in range(1, num_iter + 1):
            n_V = np.random.binomial(n, psi)  # perform n Bernoulli trials
            n_M = n - n_V
            n_M_hist[s, n - 1], n_V_hist[s, n - 1] = n_M, n_V

            # Compute the optimal policy parameters
            th_n_opt = env.opt_policy_param(n_M, n_V)
            th_n_opt_hist[s, n - 1] = th_n_opt
            if args.verbose:
                print(f"th_{n}_opt:     {th_n_opt}")

            # Compute the estimated optimal objective function value for the n domains
            Jhat_th_n_opt = env.opt_est_expec_return(n_M, n_V)
            Jhat_th_n_opt_hist[s, n - 1] = Jhat_th_n_opt
            if args.verbose:
                print(f"Jhat_{n}_opt: {Jhat_th_n_opt}")
            Jhat_n_opt_check = env.est_expec_return(th_n_opt, n_M, n_V)
            assert abs(Jhat_th_n_opt - Jhat_n_opt_check) < 1e-8

            # Check if E_\xi[max_\theta \hat{J}_n(\theta)] == max_\theta \hat{J}_n(\theta)
            if args.verbose:
                check_E_n_Jhat(th_n_opt, n)

            # Compute the estimated objective function value for the tur optimum
            Jhat_th_true_opt = env.est_expec_return(th_true_opt, n_M, n_V)
            Jhat_th_true_opt_hist[s, n - 1] = Jhat_th_true_opt

            # Create (arbitrary) candidate solutions
            noise_th = float(np.random.randn(1) * noise_th_scale)  # parameter noise
            th_c = th_true_opt + noise_th  # G_n > G_true (it should be like this)
            # th_c = th_n_opt + noise_th  # G_n < G_true (it should not be like this)
            th_c_hist[s, n - 1] = th_c
            Jhat_th_c = env.est_expec_return(th_c, n_M, n_V)
            Jhat_th_c_hist[s, n - 1] = Jhat_th_c

            # Estimated optimality gap \hat{G}_n(\theta^c)
            G_n = Jhat_th_n_opt - Jhat_th_c
            G_n_hist[s, n - 1] = G_n
            if args.verbose:
                print(f"G_{n}(th_c):\t\t{G_n}")

            # True optimality gap G(\theta^c) (use true probabilities instead of counts)
            G_true = J_true_opt - env.est_expec_return(th_c, 1 - psi, psi)
            G_true_hist[s, n - 1] = G_true
            if args.verbose:
                print(f"G_true(th_c):\t{G_true}")

            # Compute the simulation optimization bias b[\hat{J}_n]
            b_Jhat_n = calc_E_n_Jhat_th_opt(n) - J_true_opt
            b_Jhat_n_hist[s, n - 1] = b_Jhat_n
            if args.verbose:
                print(f"b_Jhat_{n}:\t\t{b_Jhat_n}\n")

    print(f"At the last iteration (n={num_iter})")
    print(f"mean G_n: {np.mean(G_n_hist, axis=0)[-1]}")
    print(f"mean G_true: {np.mean(G_true_hist, axis=0)[-1]}")
    print(f"mean b_Jhat_n: {np.mean(b_Jhat_n_hist, axis=0)[-1]}\n")

    # Plot
    os.makedirs(ex_dir, exist_ok=True)
    fig_n, ax = plt.subplots(1, figsize=fig_size, constrained_layout=True)
    draw_curve_from_data(
        "ci_on_mean",
        ax,
        n_M_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="samples per domain",
        curve_label="$n_M$",
    )
    draw_curve_from_data(
        "ci_on_mean",
        ax,
        n_V_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="samples per domain",
        curve_label="$n_V$",
    )
    ax.plot(np.arange(1, num_iter + 1), np.arange(1, num_iter + 1) * (1 - psi), c="C0", ls="--")
    ax.plot(np.arange(1, num_iter + 1), np.arange(1, num_iter + 1) * psi, c="C1", ls="--")
    ax.legend(loc="upper left", handletextpad=0.2)

    fig_theta, ax = plt.subplots(1, figsize=fig_size, constrained_layout=True)
    draw_curve_from_data(
        "ci_on_mean",
        ax,
        th_n_opt_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="policy parameter",
        curve_label=r"$\theta_n^\star$",
    )
    draw_curve_from_data(
        "ci_on_mean",
        ax,
        th_c_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="policy parameter",
        curve_label=r"$\theta^c$",
    )
    ax.plot(np.arange(1, num_iter + 1), np.ones(num_iter) * th_true_opt, ls="--", label=r"$\theta^\star$")
    ax.legend(loc="lower right", ncol=3, handletextpad=0.2)

    fig_return, ax = plt.subplots(1, figsize=fig_size, constrained_layout=True)
    draw_curve_from_data(
        "mean_std",
        ax,
        Jhat_th_n_opt_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="return",
        curve_label="$\\hat{J}_n(\\theta^\\star_n)$",
        plot_kwargs=dict(num_std=1),
    )
    draw_curve_from_data(
        "mean_std",
        ax,
        Jhat_th_c_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="return",
        curve_label="$\\hat{J}_n(\\theta^c)$",
        plot_kwargs=dict(num_std=1),
    )
    draw_curve_from_data(
        "mean_std",
        ax,
        Jhat_th_true_opt_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="return",
        curve_label="$\\hat{J}_n(\\theta^\\star)$",
        plot_kwargs=dict(num_std=1),
    )
    ax.legend(loc="lower right", ncol=3, handletextpad=0.2)
    plt.ylim(bottom=-70)

    fig_sob, ax = plt.subplots(1, figsize=fig_size, constrained_layout=True)
    draw_curve_from_data(
        "mean_std",
        ax,
        G_true_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="OG and SOB",
        curve_label=r"$G_{}^{}(\theta^c)$",
        plot_kwargs=dict(num_std=1),
    )
    draw_curve_from_data(
        "mean_std",
        ax,
        G_n_hist,
        np.arange(1, num_iter + 1),
        ax_calc=0,
        x_label="number of domains $n$",
        y_label="OG and SOB",
        curve_label=r"$\hat{G}_n^{}(\theta^c)$",
        plot_kwargs=dict(num_std=1),
    )
    ax.plot(np.arange(1, num_iter + 1), np.mean(b_Jhat_n_hist, axis=0), label=r"$\mathrm{b}[J_n(\theta^\star_n)]$")
    ax.legend(loc="upper right", ncol=3, handletextpad=0.2)
    # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), lo=3, ncol=3, mode='expand', borderaxespad=0.)
    plt.ylim(top=42)

    # Save
    if args.save:
        for fmt in ["pdf", "pgf"]:
            fig_n.savefig(osp.join(ex_dir, f"n.{fmt}"), dpi=500)
            fig_theta.savefig(osp.join(ex_dir, f"theta.{fmt}"), dpi=500)
            fig_return.savefig(osp.join(ex_dir, f"Jn.{fmt}"), dpi=500)
            fig_sob.savefig(osp.join(ex_dir, f"OG_SOB.{fmt}"), dpi=500)

    plt.show()
