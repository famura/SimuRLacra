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


def sequence(x_init, iterations, iterator_function, dtype=int):
    assert isinstance(x_init, (int, float, np.ndarray))
    assert isinstance(iterations, int) and iterations >= 0
    assert dtype == int or dtype == float

    if isinstance(x_init, np.ndarray):
        dim = len(x_init)
    else:
        dim = 1

    if iterations > 0:
        x_seq = np.zeros((iterations + 1, dim))
        x_seq[0, :] = x_init
        for i in range(1, iterations + 1):
            x_seq[i, :] = iterator_function(x_seq, i, x_init)
        # Return and type casting
        if dim == 1:
            # If x is not a np.ndarray, the last element of the sequence should also not be a np.ndarray
            return dtype(x_seq[-1, :]), x_seq.astype(dtype)
        elif dim > 1:
            return x_seq[-1, :].astype(dtype), x_seq.astype(dtype)

    else:
        if isinstance(x_init, np.ndarray):
            return x_init.copy().T, x_init.copy().T
        else:
            return x_init, x_init


def sequence_const(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0

    :param x_init: constant values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """

    def iter_function(x_seq, i, x_init):
        return x_init

    return sequence(x_init, iter, iter_function, dtype)


def sequence_plus_one(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 + n

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """

    def iter_function(x_seq, i, x_init):
        return x_seq[0, :] + i

    return sequence(x_init, iter, iter_function, dtype)


def sequence_add_init(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 * n

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    # non-exponential growth
    def iter_function(x_seq, i, x_init):
        return x_seq[0, :] * (i + 1)

    return sequence(x_init, iter, iter_function, dtype)


def sequence_rec_double(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_{n-1} * 2

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    # exponential growth
    def iter_function(x_seq, i, x_init):
        return x_seq[i - 1, :] * 2.0

    return sequence(x_init, iter, iter_function, dtype)


def sequence_sqrt(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 * sqrt(n)

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    # non-exponential growth
    def iter_function(x_seq, i, x_init):
        return x_seq[0, :] * np.sqrt(i + 1)  # i+1 because sqrt(1) = 1

    return sequence(x_init, iter, iter_function, dtype)


def sequence_rec_sqrt(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_{n-1} * sqrt(n)

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    # exponential growth
    def iter_function(x_seq, i, x_init):
        return x_seq[i - 1, :] * np.sqrt(i + 1)  # i+1 because sqrt(1) = 1

    return sequence(x_init, iter, iter_function, dtype)


def sequence_nlog2(x_init, iter, dtype=int):
    """
    Mathematical sequence: x_n = x_0 * n * log2(n+2), with log2 being the base 2 logarithm

    :param x_init: initial values of the sequence
    :param iter: iteration until the sequence should be evaluated
    :param dtype: data type to cast to (either int of float)
    :return: element at the given iteration and array of the whole sequence
    """
    # non-exponential growth
    def iter_function(x_seq, i, x_init):
        return x_seq[0, :] * i * np.log2(i + 2)  # i+2 because log2(1) = 0 and log2(2) = 1

    return sequence(x_init, iter, iter_function, dtype)
