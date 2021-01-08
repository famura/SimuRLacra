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

import os
import re

import pyrado


def filter_los_by_lok(strs: list, keys: list) -> list:
    """
    Filter a list of strings by a list of keys strings.

    :param strs: list of strings to filter
    :param keys: keys-strings to filter by
    :return: list with unique elements of the input list which contain at least one of the keys
    """
    if not isinstance(strs, list):
        raise pyrado.TypeErr(given=strs, expected_type=list)
    if not isinstance(keys, list):
        raise pyrado.TypeErr(given=keys, expected_type=list)

    # Collect all matches (multiple keys can match one string)
    all_matches = []
    for k in keys:
        all_matches.extend(list(filter(lambda s: k in s, strs)))

    # Remove non-unique element from the list
    return list(set(all_matches))


def get_immediate_subdirs(parent_dir: str):
    """
    Get all 1st level subdirectories of a specified directory (i.e. a path).

    :param parent_dir: directory in which to look for subdirectories
    :return: list of names of all 1st level subdirectories
    """
    return [f.path for f in os.scandir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]


def natural_sort(lst: list):
    """
    Sort a list like a human would do. Normal list sort does  1, 10, 11, 2, 3. But this function yields 1, 2, 3, 10, 11.

    .. seealso::
        https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort

    :param lst: input list
    :return: alpha-numerically sorted list
    """
    if not isinstance(lst, list):
        raise pyrado.TypeErr(given=lst, expected_type=list)

    def _convert(text):
        return int(text) if text.isdigit() else text.lower()

    def _alphanum_key(key):
        return [_convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(lst, key=_alphanum_key)
