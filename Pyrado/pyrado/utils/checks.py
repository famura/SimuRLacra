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


def is_iterable(obj) -> bool:
    """
    Check if the input is iterable by trying to create an iterator from the input.

    :param obj: any object
    :return: `True` if input is iterable, else `False`
    """
    try:
        _ = iter(obj)
        return True
    except TypeError:
        return False


def is_iterator(obj) -> bool:
    """
    Check if the input is an iterator by trying to call `next()` on it.

    :param obj: any object
    :return: `True` if input is an iterator, else `False`
    """
    try:
        next(obj, None)
        return True
    except TypeError:
        return False


def is_sequence(obj) -> bool:
    """
    Check if the input is a sequence.

    .. note::
        Checking if `obj` is an instance of `collections.Sequence` will return the wrong result for objects that
        implement the sequence protocol but do not inherit from `collections.Sequence`

    :param obj: any object
    :return: `True` if input is iterable, else `False`
    """
    return hasattr(type(obj), "__len__") and hasattr(type(obj), "__getitem__")


def check_all_types_equal(iterable) -> bool:
    """
    Check if all elements of an iterable are if the same type.

    :param iterable: iterable to check
    :return: bool saying if all elements are of equal type
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(type(first) == type(rest) for rest in iterator)


def check_all_lengths_equal(iterable) -> bool:
    """
    Check if the length of all elements of an iterable are equal.

    :param iterable: iterable to check
    :return: bool saying if all elements are equal
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(len(first) == len(rest) for rest in iterator)


def check_all_shapes_equal(iterable) -> bool:
    """
    Check if the shape of all elements of an iterable are equal.

    :param iterable: iterable to check
    :return: bool saying if all elements are equal
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first.shape == rest.shape for rest in iterator)


def check_all_equal(iterable) -> bool:
    """
    Check if all elements of an iterable are equal.

    :param iterable: iterable to check
    :return: bool saying if all elements are equal
    """
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True

    if isinstance(first, np.ndarray):
        return all(np.allclose(first, rest) for rest in iterator)
    else:
        return all(first == rest for rest in iterator)
