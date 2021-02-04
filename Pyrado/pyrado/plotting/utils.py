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

import math
import numpy as np
from matplotlib import colors as colors
from typing import Tuple


def num_rows_cols_from_length(n: int, transposed: bool = False) -> Tuple[int, int]:
    """
    Use a heuristic to get the number of rows and columns for a plotting grid, given the total number of plots to draw.

    :param n: total number of plots to draw
    :param transposed: change number of rows and number of columns
    :return: number of rows and columns
    """
    # First try to get a somewhat square layout
    num_rows, num_cols = most_square_product(n)
    if num_rows == 1:
        # Resort to prime factor method
        num_cols = max_prime_factor(n)  # smaller than the other factor
        num_rows = n // num_cols
    return (num_cols, num_rows) if transposed else (num_rows, num_cols)


def most_square_product(n: int) -> Tuple[int, int]:
    """
    Heuristic to get two square-like integers that when multiplied yield the input

    :param n: given number $n$
    :return: lower and higher integer
    """
    o = math.ceil(math.sqrt(n))
    m = o - 1
    if o ** 2 == n:
        return o, o
    else:
        i = 0
        while o * m != n and m != 1:
            if i % 2 == 0:
                # Every even iteration
                o += 1
            else:
                # Every odd iteration
                m -= 1
            # Increase iteration count
            i += 1
        return m, o


def max_prime_factor(n: int) -> int:
    r"""
    Get the largest prime number that is a factor of the given number

    .. seealso::
        https://www.w3resource.com/python-exercises/challenges/1/python-challenges-1-exercise-35.php

    :param n: given number $n$
    :return: largest prime number $p$ such that $p \cdot x = n$
    """
    prime_factor = 1
    i = 2

    while i <= n / i:
        if n % i == 0:
            prime_factor = i
            n /= i
        else:
            i += 1

    if prime_factor < n:
        prime_factor = n

    return int(prime_factor)


class AccNorm(colors.Normalize):
    """
    Accumulative normalizer which is useful to have one colormap consistent for multiple images.
    Adding new data will expand the limits.
    """

    def autoscale(self, A):
        # Also update values if scale expands
        vmin = np.min(A)
        if self.vmin is None or self.vmin > vmin:
            self.vmin = vmin

        vmax = np.max(A)
        if self.vmax is None or self.vmax < vmax:
            self.vmax = vmax

    def autoscale_None(self, A):
        # Also update values if scale expands
        vmin = np.min(A)
        if self.vmin is None or self.vmin > vmin:
            self.vmin = vmin

        vmax = np.max(A)
        if self.vmax is None or self.vmax < vmax:
            self.vmax = vmax
