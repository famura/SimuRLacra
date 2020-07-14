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

import torch as to
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

import pyrado


def sample_from_hyper_sphere_surface(num_dim: int, method: str) -> to.Tensor:
    """
    Sampling from the surface of a multidimensional unit sphere.

    .. seealso::
        [1] G. Marsaglia, "Choosing a Point from the Surface of a Sphere", Ann. Math. Statist., 1972

    :param num_dim: number of dimensions of the sphere
    :param method: approach used to acquire the samples
    :return: sample with L2-norm equal 1
    """
    assert num_dim > 0
    num_dim = int(num_dim)

    if method == 'uniform':
        # Initialization
        ones = to.ones((num_dim,))
        udistr = Uniform(low=-ones, high=ones)
        sum_squares = pyrado.inf

        # Sample candidates until criterion is met
        while sum_squares >= 1:
            sample = udistr.sample()
            sum_squares = sample.dot(sample)

        # Return scaled sample
        return sample/to.sqrt(sum_squares)

    elif method == 'normal':
        # Sample fom standardized normal
        sample = Normal(loc=to.zeros((num_dim,)), scale=to.ones((num_dim,))).sample()

        # Return scaled sample
        return sample/to.norm(sample, p=2)

    elif method == 'Marsaglia':
        if not (num_dim == 3 or num_dim == 4):
            raise pyrado.ValueErr(msg="Method 'Marsaglia' is only defined for 3-dim space")
        else:
            # Initialization
            ones = to.ones((2,))
            udistr = Uniform(low=-ones, high=ones)
            sum_squares = pyrado.inf

            # Sample candidates until criterion is met
            while sum_squares >= 1:
                sample = udistr.sample()
                sum_squares = sample.dot(sample)

            if num_dim == 3:
                # Return scaled sample
                return to.tensor([2*sample[0]*to.sqrt(1 - sum_squares),
                                  2*sample[1]*to.sqrt(1 - sum_squares),
                                  1 - 2*sum_squares])
            else:
                # num_dim = 4
                sum_squares2 = pyrado.inf
                while sum_squares2 >= 1:
                    sample2 = udistr.sample()
                    sum_squares2 = sample2.dot(sample2)
                # Return scaled sample
                return to.tensor([sample[0], sample[1],
                                  sample2[0]*to.sqrt((1 - sum_squares)/sum_squares2),
                                  sample2[1]*to.sqrt((1 - sum_squares)/sum_squares2)])
    else:
        raise pyrado.ValueErr(given=method, eq_constraint="'uniform', 'normal', or 'Marsaglia'")
