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
from typing import Sequence

from pyrado.spaces import BoxSpace


class Polar2DPosSpace(BoxSpace):
    """
    Samples positions on a 2-dim torus, i.e. the area between two concentric circles.
    Can also be a section of a 2-dim torus, i.e. not a full circle.
    """

    def __init__(self,
                 bound_lo: [float, list, np.ndarray],
                 bound_up: [float, list, np.ndarray],
                 shape: [tuple, int] = None,
                 labels: Sequence[str] = None):
        """
        Constructor

        :param bound_lo: minimal distance and the minimal angle (polar coordinates)
        :param bound_up: maximal distance and the maximal angle (polar coordinates)
        :param shape: tuple specifying the shape, useful if all lower and upper bounds are identical
        :param labels: label for each dimension of the space
        """
        assert bound_lo.size == bound_up.size == 2
        # Actually, this space is a BoxSpace
        super().__init__(bound_lo, bound_up, shape, labels=labels)

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Get a random sample from the polar space
        sample = super().sample_uniform()
        # Transform the positions to the cartesian space
        return np.array([sample[0]*np.cos(sample[1]), sample[0]*np.sin(sample[1])])

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        assert cand.size == 2
        # Transform candidate to polar space
        x, y = cand[0], cand[1]
        polar = np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)])  # arctan2 returns in range [-pi, pi] -> check bounds
        # Query base
        return super().contains(polar, verbose=verbose)


class Polar2DPosVelSpace(BoxSpace):
    """
    Samples positions on a 2-dim torus, i.e. the area between 2 circles augmented with cartesian velocities.
    Can also be a section of a 2-dim torus, i.e. not a full circle.
    """

    def __init__(self,
                 bound_lo: [float, list, np.ndarray],
                 bound_up: [float, list, np.ndarray],
                 shape: [tuple, int] = None,
                 labels: Sequence[str] = None):
        """
        Constructor

        :param bound_lo: minimal distance, the minimal angle, and the 2D minimal cartesian initial velocity
        :param bound_up: maximal distance, the maximal angle, and the 2D minimal cartesian initial velocity
        :param shape: tuple specifying the shape, useful if all lower and upper bounds are identical
        :param labels: label for each dimension of the space
        """
        assert bound_lo.size == bound_up.size == 4
        # Actually, this space is a BoxSpace
        super().__init__(bound_lo, bound_up, shape, labels=labels)

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Get a random sample from the half-polar / half-cartesian space
        sample = super().sample_uniform()
        # Transform the positions to the cartesian space
        sample[:2] = np.array([sample[0]*np.cos(sample[1]), sample[0]*np.sin(sample[1])])
        return sample

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        assert cand.size == 4
        # Transform candidate to polar space
        x, y = cand[0], cand[1]
        polar = np.array([np.sqrt(x**2 + y**2), np.arctan2(y, x)])  # arctan2 returns in range [-pi, pi] -> check bounds
        # Query base
        return super().contains(np.r_[polar, cand[2:]], verbose=verbose)
