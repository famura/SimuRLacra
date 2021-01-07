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
Script to play around with normalizing flows

.. note::
    https://github.com/bayesiains/nflows/blob/master/examples/moons.ipynb
"""
import torch as to
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import RandomPermutation
from nflows.transforms.permutations import ReversePermutation

from pyrado.policies.feed_forward.nflow import NFlowPolicy


if __name__ == "__main__":
    num_layers = 5
    base_dist = StandardNormal(shape=[2])

    transforms = []
    for _ in range(num_layers):
        # From the readme
        # transforms.append(MaskedAffineAutoregressiveTransform(features=2, hidden_features=4))
        # transforms.append(RandomPermutation(features=2))
        # From the example (subjectively works better)
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2, hidden_features=4))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    mapping = {0: "dp1", 1: "dp2"}
    trafo_mask = [False, False]
    policy = NFlowPolicy(flow, mapping, trafo_mask)
    print(f"number of policy parameters {policy.num_param}")
    optimizer = optim.Adam(policy.parameters())

    num_iter = 5000
    for i in range(num_iter):
        x, y = datasets.make_moons(256, noise=0.1)
        x = to.tensor(x, dtype=to.float32)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            xline = to.linspace(-1.5, 2.5, 100)
            yline = to.linspace(-0.75, 1.25, 100)
            xgrid, ygrid = to.meshgrid(xline, yline)
            xyinput = to.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

            with to.no_grad():
                zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy())
            plt.title("iteration {}".format(i + 1))
            plt.show()
