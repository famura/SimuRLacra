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
A small collection of well-known functions for testing or benchmarking
"""
import numpy as np
import torch as to

import pyrado


def rosenbrock(x: [to.Tensor, np.ndarray]) -> (to.Tensor, np.ndarray):
    """
    The Rosenbrock function
    (consistent with https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html)
    :param x: multi-dim column vector, or array thereof
    :return: value of the rosenbrock function at the input point, or array thereof
    """
    if isinstance(x, to.Tensor):
        return to.sum(100.*to.pow(x[1:] - to.pow(x[:-1], 2), 2) + to.pow((1. - x[:-1]), 2), dim=0)
    elif isinstance(x, np.ndarray):
        return np.sum(100.*np.power(x[1:] - np.power(x[:-1], 2), 2) + np.power((1. - x[:-1]), 2), axis=0)
    else:
        raise pyrado.TypeErr(given=x, expected_type=[np.ndarray, to.Tensor])


def noisy_nonlin_fcn(x: [to.Tensor, np.ndarray], f: float = 1., noise_std: float = 0.) -> [to.Tensor, np.ndarray]:
    """
    A 1-dim function (sinus superposed with polynomial), representing the black box function in Bayesian optimization

    :param x: function argument
    :param noise_std: scale of the additive noise sampled from a standard normal distribution
    :param f: frequency of the sinus wave [Hz]
    :return: function value
    """
    if isinstance(x, to.Tensor):
        return -to.sin(2*np.pi*f*x) - to.pow(x, 2) + 0.7*x + noise_std*to.randn_like(x)
    elif isinstance(x, np.ndarray):
        return -np.sin(2*np.pi*f*x) - np.power(x, 2) + 0.7*x + noise_std*np.random.randn(*x.shape)
    else:
        raise pyrado.TypeErr(given=x, expected_type=[np.ndarray, to.Tensor])
