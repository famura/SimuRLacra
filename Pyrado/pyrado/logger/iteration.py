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

from contextlib import contextmanager


class IterationTracker:
    """ Track the current iteration/step number on multiple levels (for meta-algorithms) """

    def __init__(self):
        """ Constructor """
        self._iter_stack = []

    def push(self, label: str, num: int):
        """
        Push an iteration scope.

        :param label: scope label
        :param num: iteration index
        """
        self._iter_stack.append((label, num))

    def pop(self) -> tuple:
        """ Remove the last iteration scope. """
        return self._iter_stack.pop()

    def peek(self) -> tuple:
        return self._iter_stack[-1]

    @contextmanager
    def iteration(self, label: str, num: int):
        """
        Context with active iteration scope.

        :param label: scope label
        :param num: iteration index
        """
        self.push(label, num)
        yield
        self.pop()

    def get(self, label: str):
        """
        Get the iteration number for a labeled scope.

        :param label: scope label
        :return: iteration index
        """
        for l, n in self._iter_stack:
            if l == label:
                return n
        return None

    def __iter__(self):
        yield from self._iter_stack

    def format(self, scope_sep="-", label_num_sep="_"):
        """
        Format the current iteration stack into a string. Two parts can be customized:

        :param scope_sep: string separating the label and the number
        :param label_num_sep: string separating each label/number pair
        :return: string with custom separators
        """
        return scope_sep.join(l + label_num_sep + str(n) for l, n in self._iter_stack)

    def __str__(self):
        """ Get an information string. """
        return self.format()
