import torch as to
from torch import nn
from torch import optim

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

num_layers = 5
base_dist = StandardNormal(shape=[2])

transforms = []

def train_nflows(samples, num_iter = 50, features = 2):
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=4))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    optimizer = optim.Adam(flow.parameters())

    print()
    loss = "None"
    for i in range(num_iter):
        if not i % 10:
            print("\r[NFLOWS] Iteration ({}|{}); Loss: {}".format(i, num_iter, loss), end='')
        # x = to.tensor(samples, dtype=to.float32)
        x = samples.clone().detach()
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x).mean()
        loss.backward()
        optimizer.step()
    print("\n[NFLOWS] Traning ended")

    return flow
