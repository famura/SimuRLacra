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
Script to export a PyTorch-based Pyrado policy to C++
"""
import numpy as np
import torch as to
from rcsenv import ControlPolicy

from pyrado.policies.features import FeatureStack, const_feat, identity_feat, squared_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.policies.recurrent.rnn import RNNPolicy
from pyrado.spaces.box import BoxSpace
from pyrado.utils.data_types import EnvSpec


def create_nonrecurrent_policy():
    return LinearPolicy(
        EnvSpec(
            BoxSpace(-1, 1, 4),
            BoxSpace(-1, 1, 3),
        ),
        FeatureStack([const_feat, identity_feat, squared_feat]),
    )


def create_recurrent_policy():
    return RNNPolicy(
        EnvSpec(
            BoxSpace(-1, 1, 4),
            BoxSpace(-1, 1, 3),
        ),
        hidden_size=32,
        num_recurrent_layers=1,
        hidden_nonlin="tanh",
    )


if __name__ == "__main__":
    tmpfile = "/tmp/torchscriptsaved.pt"
    to.set_default_dtype(to.float32)  # former double

    # Create a Pyrado policy
    model = create_nonrecurrent_policy()
    # model = create_recurrent_policy()

    # Trace the Pyrado policy (inherits from PyTorch module)
    traced_script_module = model.script()
    print(traced_script_module.graph)

    # Save the scripted module
    traced_script_module.save(tmpfile)

    # Load in C++
    cp = ControlPolicy("torch", tmpfile)

    # Print more digits
    to.set_printoptions(precision=8, linewidth=200)
    np.set_printoptions(precision=8, linewidth=200)

    print(f"manual: {model(to.tensor([1, 2, 3, 4], dtype=to.get_default_dtype()))}")
    print(f"script: {traced_script_module(to.tensor([1, 2, 3, 4], dtype=to.get_default_dtype()))}")
    print(f"cpp:    {cp(np.array([1, 2, 3, 4]), 3)}")
