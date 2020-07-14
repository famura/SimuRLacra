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
import torch.nn as nn
from tqdm import tqdm

from pyrado.logger.step import StepLogger
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


device = to.device('cuda' if to.cuda.is_available() else 'cpu')


class RewardGenerator:
    """
    Class for generating the discriminator rewards in ADR. Generates a reward using a trained discriminator network.
    """

    def __init__(self,
                 env_spec: EnvSpec,
                 batch_size: int,
                 reward_multiplier: float,
                 lr: float = 3e-3,
                 logger: StepLogger = None):
        """
        Constructor

        :param env_spec: environment specification
        :param batch_size: batch size for each update step
        :param reward_multiplier: factor for the predicted probability
        :param lr: learning rate
        :param logger: logger instance
        """
        self.batch_size = batch_size
        self.reward_multiplier = reward_multiplier
        self.lr = lr
        # self.discriminator = LSTMDiscriminator(env_spec.obs_space.flat_dim, env_spec.act_space.flat_dim, batch_size)
        self.discriminator = MLPDiscriminator(env_spec.obs_space.flat_dim, env_spec.act_space.flat_dim)
        self.loss_fcn = nn.BCELoss()
        self.optimizer = to.optim.Adam(self.discriminator.parameters(), lr)
        self.logger = logger

    def step(self, snapshot_mode: str, meta_info: dict = None):
        pass

    def get_reward(self, traj: StepSequence):
        traj = convert_step_sequence(traj)
        with to.no_grad():
            reward = self.discriminator.forward(traj).cpu()
            return to.log(reward.mean())*self.reward_multiplier

    def train(self,
              reference_trajectory: StepSequence,
              randomized_trajectory: StepSequence,
              num_epoch: int) -> to.Tensor:

        reference_batch = reference_trajectory.split_shuffled_batches(self.batch_size)
        random_batch = randomized_trajectory.split_shuffled_batches(self.batch_size)

        for _ in tqdm(range(num_epoch), 'Discriminator Epoch', num_epoch):
            try:
                reference_batch_now = convert_step_sequence(next(reference_batch))
                random_batch_now = convert_step_sequence(next(random_batch))
            except StopIteration:
                break
            if reference_batch_now.shape[0] < self.batch_size - 1 or random_batch_now.shape[0] < self.batch_size - 1:
                break
            random_results = self.discriminator(random_batch_now)
            reference_results = self.discriminator(reference_batch_now)
            self.optimizer.zero_grad()
            loss = self.loss_fcn(random_results,
                                 to.ones(self.batch_size - 1)) + self.loss_fcn(reference_results,
                                                                               to.zeros(self.batch_size - 1))
            loss.backward()
            self.optimizer.step()

            # Logging
            if self.logger is not None:
                self.logger.add_value('discriminator_loss', loss)
        return loss


class MLPDiscriminator(nn.Module):
    """ MLP-based discriminator """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        """
        Constructor
        
        :param obs_dim: observation space dimension
        :param act_dim: action space dimension
        :param hidden_dim: hidden layer size
        """
        super(MLPDiscriminator, self).__init__()
        assert isinstance(hidden_dim, int)
        head = nn.Linear(hidden_dim, 1)
        head.weight.data.mul_(0.1)
        head.bias.data.mul_(0.1)

        self.net = nn.Sequential(
            nn.Linear((2*obs_dim + act_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            head,
            nn.Sigmoid(),
        )

    def forward(self, x):
        """

        :param x: A Tensor which contains the state, action and next state
        :return: The predicted probability that this tensor originates from a randomized environment
        """
        return self.net(x)


class LSTMDiscriminator(nn.Module):
    """ LSTM-based discriminator """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        """
        Constructor

        :param obs_dim: observation space dimension
        :param act_dim: action space dimension
        :param hidden_dim: hidden layer size
        """
        super(LSTMDiscriminator, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = 2

        input_dim = 2*obs_dim + act_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, self.layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """

        :param x: A Tensor which contains the state, action and next state
        :return: The predicted probability that this tensor originates from a randomized environment
        """
        x = x.unsqueeze(0)

        h0 = to.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        c0 = to.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.fc(out[:, -1, :])
        return to.sigmoid(out)


def convert_step_sequence(traj: StepSequence):
    """
    Converts a StepSequence to a Tensor which can be fed through a Network

    :param traj: A step sequence containing a trajectory
    :return: A Tensor containing the trajectory
    """
    assert isinstance(traj, StepSequence)
    traj.torch()
    state = traj.get_data_values('observations')[:-1].double()
    next_state = traj.get_data_values('observations')[1::].double()
    action = traj.get_data_values('actions').narrow(0, 0, next_state.shape[0]).double()
    traj = to.cat((state, next_state, action), 1).cpu().double()
    return traj
