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
Script to export a recurrent PyTorch-based policy to C++
"""
import numpy as np
import torch as to
from rcsenv import ControlPolicy
from torch.jit import script

from pyrado.policies.recurrent.adn import ADNPolicy, pd_capacity_21
from pyrado.policies.recurrent.base import StatefulRecurrentNetwork
from pyrado.policies.recurrent.rnn import RNNPolicy
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    tmpfile = "/tmp/torchscriptsaved.pt"
    to.set_default_dtype(to.float32)  # former double

    # Seclect the policy type
    policy_type = "RNN"

    if policy_type == "RNN":
        net = RNNPolicy(
            EnvSpec(
                BoxSpace(-1, 1, 4),
                BoxSpace(-1, 1, 2),
            ),
            hidden_size=10,
            num_recurrent_layers=2,
        )
    elif policy_type == "ADN":
        net = ADNPolicy(
            EnvSpec(
                BoxSpace(-1, 1, 4),
                BoxSpace(-1, 1, 2),
            ),
            dt=0.01,
            activation_nonlin=to.sigmoid,
            potentials_dyn_fcn=pd_capacity_21,
        )
    else:
        raise NotImplementedError

    # Trace the policy
    #     traced_net = trace(net, (to.from_numpy(net.env_spec.obs_space.sample_uniform()), net.init_hidden()))
    #     print(traced_net.graph)
    #     print(traced_net(to.from_numpy(net.env_spec.obs_space.sample_uniform()), None))

    stateful_net = script(StatefulRecurrentNetwork(net))
    print(stateful_net.graph)
    print(stateful_net.reset.graph)
    print(list(stateful_net.named_parameters()))

    stateful_net.save(tmpfile)

    # Load in c
    cp = ControlPolicy("torch", tmpfile)

    inputs = [
        [1.0, 2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0, 6.0],
    ]

    hid_man = net.init_hidden()
    for inp in inputs:
        # Execute manually
        out_man, hid_man = net(to.tensor(inp), hid_man)
        # Execute script
        out_sc = stateful_net(to.tensor(inp))
        # Execute C++
        out_cp = cp(np.array(inp), 2)

        print(f"{inp} =>")
        print(f"manual: {out_man}")
        print(f"script: {out_sc}")
        print(f"cpp:    {out_cp}")
