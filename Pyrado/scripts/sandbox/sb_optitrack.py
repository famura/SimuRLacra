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
Script to test the NatNet client in combination with OptiTrack
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from pyrado.environments.barrett_wam.natnet_client import NatNetClient
from pyrado.environments.barrett_wam.trackers import RigidBodyTracker


if __name__ == "__main__":
    # OptiTrack client
    streamingClient = NatNetClient(ver=(3, 0, 0, 0), quiet=True)
    rbdt = RigidBodyTracker(
        ["Cup", "Ball"],
        # rotation=Rotation.from_euler("XYZ", [90.0, -90.0, 0.0], degrees=True),  # same as below
        rotation=Rotation.from_euler("yxz", [-90.0, 90.0, 0.0], degrees=True),  # same as above
        offset=np.array([-0.4220, 0.4140, -0.1097]),
    )
    streamingClient.rigidBodyListener = rbdt

    # Start data streaming
    streamingClient.run()

    while not rbdt.initialized():
        time.sleep(0.05)

    for i in range(0, 10000):
        # print(rbdt.get_current_estimate(["Cup", "Ball"]))
        print(rbdt.get_current_estimate(["Ball"])[0])
        time.sleep(1)

    streamingClient.stop()
