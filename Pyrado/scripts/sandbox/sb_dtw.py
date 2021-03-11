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
Test dynamic time warping implementation from the dtw-python package
"""
import numpy as np
from dtw import *

if __name__ == "__main__":
    # A noisy sine wave as query
    idx = np.linspace(0, 6.28, num=100)
    multidim = False

    if multidim:
        query = np.stack([np.sin(idx), np.sin(idx)], axis=1)
        template = np.stack([np.cos(idx), np.cos(idx)], axis=1)
    else:
        query = np.sin(idx)  # + np.random.uniform(size=100) / 10.0
        template = np.cos(idx)  # sin and cos are offset by 25 samples

    # Find the best match with the canonical recursion formula
    alignment = dtw(query, template, keep_internals=True)

    # Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    alignment2 = dtw(query, template, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))

    # Display the warping curve, i.e. the alignment curve
    if not multidim:
        alignment.plot(type="twoway")
        alignment2.plot(type="twoway")

    print(f"distance symmetric2: {alignment.distance}\ndistance rabinerJuang: {alignment2.distance}")
