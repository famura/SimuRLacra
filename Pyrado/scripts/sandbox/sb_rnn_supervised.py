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
import torch as to
import torch.optim as optim
import torch.nn as nn

from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from matplotlib import pyplot as plt
from pyrado.policies.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado import set_seed


if __name__ == '__main__':
    # -----
    # Setup
    # -----

    # Generate the data
    set_seed(1001)
    env = OneMassOscillatorSim(dt=0.01, max_steps=500)
    ro = rollout(env, IdlePolicy(env.spec), reset_kwargs={'init_state': np.array([0.5, 0.])})
    ro.torch(data_type=to.get_default_dtype())
    inp = ro.observations[:-1, 0] + 0.01*to.randn(ro.observations[:-1, 0].shape)  # added observation noise
    targ = ro.observations[1:, 0]

    # Problem dimensions
    inp_size = 1
    targ_size = 1
    num_trn_samples = inp.shape[0]

    # Hyper-parameters
    loss_fcn = nn.MSELoss()
    num_epoch = 1000
    num_layers = 1
    hidden_size = 20  # targ_size
    lr = 1e-3

    # Create the recurrent neural network
    net = nn.RNN(
        input_size=inp_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity='tanh',
        # batch_first=True
    )

    # Create the optimizer
    optim = optim.Adam([{'params': net.parameters()}], lr=lr, eps=1e-8, weight_decay=1e-5)

    # --------
    # Training
    # --------

    # Iterations over the whole data set
    for e in range(num_epoch):
        # Reset the gradients and the hidden states
        optim.zero_grad()
        hidden = to.zeros(num_layers, 1, hidden_size)

        # Make predictions (complete trajectory)
        output, _ = net(inp.view(num_trn_samples, 1, -1), hidden)
        # Compute policy loss (complete trajectory)
        loss = loss_fcn(targ, output[:, 0, -targ_size])

        # Call optimizer
        loss.backward()
        optim.step()

        if e%10 == 0:
            loss_avg = loss.item()/num_trn_samples
            print(f'Epoch {e:4d} | avg loss {loss_avg:.3e}')

    # -------
    # Testing
    # -------

    pred = []
    informative_hidden_init = False
    num_init_steps = num_layers*hidden_size

    hidden = to.zeros(num_layers, 1, hidden_size)
    if informative_hidden_init:
        output, hidden = net(inp[:num_init_steps].view(num_init_steps, 1, -1), hidden)

    for i in range(int(informative_hidden_init)*num_init_steps, num_trn_samples):
        output, hidden = net(inp[i].view(1, 1, -1), hidden)
        pred.append(output[:, :, -targ_size].view(-1))

    # Plotting
    pred = np.array(pred)
    targ = targ[int(informative_hidden_init)*num_init_steps:].numpy()
    inp = inp[int(informative_hidden_init)*num_init_steps:].numpy()
    plt.plot(targ, label='target')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()
