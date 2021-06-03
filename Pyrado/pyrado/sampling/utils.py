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

import random
from itertools import islice
from math import ceil
from typing import Iterable


def gen_shuffled_batch_idcs(batch_size: int, data_size: int):
    """
    Helper function for doing SGD on mini-batches that returns the indies for the mini-batch samples

    :param batch_size: number of samples in each mini-batch
    :param data_size: total number of samples
    :return: generator for lists of random indices of sub-samples

    :usage:

        If num_rollouts = 2 and data_size = 5, then the output might be ((0, 3), (2, 1), (4,))
    """
    # random.seed(0)
    idx_all = random.sample(range(data_size), data_size)
    idx_iter = iter(idx_all)
    return iter(lambda: list(islice(idx_iter, batch_size)), [])


def gen_ordered_batch_idcs(batch_size: int, data_size: int, sorted: bool = True):
    """
    Helper function for doing SGD on mini-batches that returns the indies for the mini-batch samples

    :param batch_size: number of samples in each mini-batch
    :param data_size: total number of samples
    :param sorted: if `False`, the order of batches is randomized (but the order within them is preserved)
    :return: generator for lists of random indices of sub-samples

    :usage:

        If num_rollouts = 2, data_size = 5 and sorted = False, then the output might be ((2, 3), (0, 1), (4,))
        If num_rollouts = 2, data_size = 5 and sorted = True, then the output will be ((0, 1), (2, 3), (4,))
    """
    num_batches = int(ceil(data_size / batch_size))

    # Create a list of lists, each containing num_rollouts ordered elements
    idcs_all = list(range(data_size))
    idcs_batches = [idcs_all[i * batch_size : i * batch_size + batch_size] for i in range(num_batches)]

    if not sorted:
        # Yield a random sample from the list of lists
        idcs_batches = random.sample(idcs_batches, len(idcs_batches))

    idx_iter = iter(idcs_batches)
    return iter(islice(idx_iter, num_batches))


def gen_ordered_batches(inp: Iterable, batch_size: int):
    """
    Helper function that cuts the input into equal sized chunks

    :param inp: iterable input
    :param batch_size: number of samples in each mini-batch
    :return: iterator over the input
    """
    it = iter(inp)
    return iter(lambda: list(islice(it, batch_size)), [])


def shuffled_ordered_batches(inp: Iterable, batch_size: int):
    """
    Helper function that cuts the input into equal sized chunks with the original ordering, but shuffled order among the
    chunks

    :param inp: iterable input
    :param batch_size: number of samples in each mini-batch
    :return: list of randomly ordered mini-batches which within themselves have the original ordering
    """
    ordered_batches = list(gen_ordered_batches(inp, batch_size))
    return random.sample(ordered_batches, len(ordered_batches))
