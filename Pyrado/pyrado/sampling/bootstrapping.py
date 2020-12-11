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
from warnings import warn

import pyrado


def bootstrap_ci(
    data: np.ndarray,
    stat_fcn,
    num_reps: int,
    alpha: float,
    ci_sides: int,
    bias_correction: bool = False,
    studentized: bool = False,
    seed: int = None,
):
    r"""
    Re-sampling input data using the nonparametric bootstrap method, computing bootstrap replications using stat_fcn and
    computing a confidence interval on the statistic of interest given by stat_fcn.

    .. seealso::
        [1] https://projecteuclid.org/download/pdf_1/euclid.ss/1032280214
        [2] https://people.csail.mit.edu/tommi/papers/SteJaa-nips03.pdf
        [3] Cameron & Trivedi, "Microeconometrics: Methods and Applications", 2005, page 361
        [4] http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
        [5] https://www.diva-portal.org/smash/get/diva2:130905/FULLTEXT01.pdf
        [6] https://www.ethz.ch/content/dam/ethz/special-interest/math/statistics/sfs/Education/Advanced%20Studies%20in%20Applied%20Statistics/course-material-1719/Nonparametric%20Methods/lecture_2up.pdf
        [7] https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf

    :param data: data to bootstrap from (for now only 1D arrays supported)
    :param stat_fcn: function to compute a statistic of interest (e.g. mean, variance) on bootstrap samples
    :param num_reps: number of samples in every bootstrap sample
    :param alpha: determines the confidence level $1 - \alpha \in [0, 1]$
    :param ci_sides: one or two-sided confidence interval
    :param bias_correction: bool to decide if the bias should be subtracted (see [2]). However, the confidence intervals
                            are constructed independent of the bias-correction (see [5, p.7])
    :param studentized: flag to determine if the method based on the t-distribution is used (leads to a wider ci)
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return: mean of the bootstrap replications and the confidence interval
    """
    assert isinstance(data, np.ndarray)
    assert isinstance(num_reps, int) and num_reps > 0
    assert ci_sides == 1 or ci_sides == 2

    # Compute the statistic of interest based on the empirical distribution (input data)
    stat_emp = stat_fcn(data)

    # Get the bootstrap replications
    if data.ndim == 1:
        # Size of the samples drawn by the bootstrap method have to be equal input sample, since the variance of the
        # statistic to be computed depends on sample size
        size_sample = data.shape[0]
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)
        # Draw samples of data with replacement (uniform weights)
        data_bs = np.random.choice(data, (size_sample, num_reps), replace=True)
    else:
        # Only use this function for 1D data sets
        raise NotImplementedError

    # Compute the statistic of interest based on the resampled distribution
    # Do it along each row (axis=0) -->> bootstrap replications
    stat_bs = np.apply_along_axis(stat_fcn, 0, data_bs)  # dim = 1 x num_reps

    # Correct for the bias introduced by bootstrapping
    # Note: other estimates of the bias-correction factor than stat_emt possible, see [4]
    if bias_correction:
        # bias-corrected statistic (see (2) in [2], or (11.10) in [3])
        stat_bs_bc = 2 * stat_emp - np.mean(stat_bs)  # same as bias = mean_repl - stat_emp; repl_bc = stat_emp - bias
        stat_ret = stat_bs_bc  # use the correction also for the bs replications? -->> No (so far)
        # Note: bias-correction can be dangerous in practice. Even though T_bc(D) is less biased than T(D),
        # the bias-corrected estimator may have substantially larger variance. This is due to a possibly higher
        # variability in the estimate of the bias, particularly when computed from small data sets.
    else:
        # Return the estimator based on the original sample a.k.a. empirical distribution
        stat_ret = stat_emp

    # Compute the deviation to the value of the statistic based on the empirical distribution (see [7])
    # This is analogous to the deviation of the empirical value around the true population value
    # i.e. delta = stat_emp - stat_pop
    # Note: it makes no difference if one uses the percentile operator before or after this difference
    delta_bs = stat_bs - stat_emp  # dim = 1 x num_reps

    # Confidence interval with asymptotic refinement (a.k.a. percentile-t method)
    if studentized:
        # Compute the standard deviation of the original sample
        se_emp = np.std(data, ddof=0) / np.sqrt(data.shape[0])  # for dividing by (n-1) set ddof=1
        if se_emp < 1e-9:
            warn("Standard deviation in the empirical data (se_emp) is below 1e-9.", UserWarning)

        # Compute the standard error of the replications for the bootstrapped t-statistic
        se_bs = np.std(stat_bs, ddof=0) / np.sqrt(data_bs.shape[0])  # dim = num_reps x 1
        if se_bs < 1e-9:  # use any for version 2 above
            warn(
                "Standard deviation in the bootstrapped data (se_bs) is below 1e-9. "
                "Setting confidence interval bounds to infinity.",
                UserWarning,
            )
            return stat_ret, [-np.infty, np.infty]

        # Compute the t-statistic of the replications
        t_bs = delta_bs / se_bs  # is consistent with [3, p. 360]

        if ci_sides == 2:  # Two-sided confidence interval
            t_bs.sort()
            t_lo, t_up = np.percentile(t_bs, [100 * alpha / 2.0, 100 - 100 * alpha / 2.0])
            ci_lo = stat_emp - t_up * se_emp  # see [3, (11.6) p. 364]
            ci_up = stat_emp - t_lo * se_emp  # see [3, (11.6) p. 364]

        elif ci_sides == 1:  # One-sided confidence interval (upper bound)
            t_bs.sort()
            t_lo = np.percentile(t_bs, 100 * alpha)
            ci_lo = -np.inf
            ci_up = stat_emp - t_lo * se_emp  # see [3, (11.6) p. 364]

        else:
            raise pyrado.ValueErr(given=ci_sides, eq_constraint="1 or 2")

    # Confidence interval without asymptotic refinement (a.k.a. basic method)
    else:
        if ci_sides == 2:  # Two-sided confidence interval
            delta_bs.sort()
            delta_lo, delta_up = np.percentile(delta_bs, [100 * alpha / 2.0, 100 - 100 * alpha / 2.0])
            ci_lo = stat_emp - delta_up
            ci_up = stat_emp - delta_lo

        elif ci_sides == 1:  # One-sided confidence interval (upper bound)
            delta_bs.sort()
            delta_lo = np.percentile(delta_bs, 100 * alpha)
            ci_lo = -np.inf
            ci_up = stat_emp - delta_lo

        else:
            raise pyrado.ValueErr(given=ci_sides, eq_constraint="1 or 2")

    return stat_ret, [ci_lo, ci_up]
