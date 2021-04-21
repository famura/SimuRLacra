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

from threading import Lock
from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation


class RigidBodyTracker:
    """
    Class for tracking rigid bodies with OptiTrack and a NatNet client
    This code is based on work from Pascal Klink.
    """

    def __init__(self, names: Sequence[str], rotation=None, offset: Union[np.ndarray, list] = np.zeros(3)):
        """
        Constructor

        :param names: list of rigid body names, e.g. ["cup", "ball"]
        :param rotation: `Rotation` instance of scipy.spatial.transform
        :param offset: [x, y, z] offset from the OptiTrack coordinate system to the one used in simulation
        """
        self.names = names
        self.rotation = rotation
        self.offset = np.asarray(offset)
        self.names_map = {}
        self.ts = {}
        self.lock = Lock()

    def __call__(self, rb_id, raw_name, pos, rot):
        """
        The implementation of callback function `rigidBodyListener` of the NatNet client.

        :param rb_id: ID of the rigid body
        :param raw_name: name of the rigid body
        :param pos: cartesian position
        :param rot: rotation specified as quaternion
        """
        name = raw_name.decode("utf-8")
        if name not in self.names:
            return

        # Add body to names_map if it is not yet added
        if name not in self.names_map:
            self.names_map[name] = rb_id

        if self.names_map[name] != rb_id:
            raise RuntimeError("Rigid Body ID changed during tracking!")

        # Save data as tuple, OptiTrack streams in xyzw (quaternion)
        self.ts[name] = (pos, Rotation.from_quat(rot))

    def reset_offset(self):
        """ Reset the Cartesian offset, e.g. before calibrating. """
        self.offset = np.zeros(3)

    def initialized(self):
        """ Check if all rigid bodies have been seen at least once. """
        return len(self.names) == len(self.names_map)

    def get_current_estimate(self, names: Sequence[str]) -> [List[Tuple], Tuple]:
        """
        Get position and rotation estimate for the given body names.

        :param names: list of rigid body names
        :return copied_ts: list of tuples containing the current estimate for each body in names, or tuple if there
                           would only be one element in the list
        """
        copied_ts = []
        self.lock.acquire()
        for name in names:
            if name not in self.ts:
                raise RuntimeError(f"Given Name {name} not in the position map")

            t = self.ts[name]

            # Apply rotation (e.g. to MuJoCo frame) if given, then shift the offset
            if self.rotation is not None:
                copied_ts.append((self.rotation.apply(t[0]) - self.offset, self.rotation * t[1]))
            else:
                copied_ts.append((t[0], t[1]))

        self.lock.release()
        return copied_ts if len(copied_ts) > 1 else copied_ts[0]


class MarkerTracker:
    """
    Class for tracking markers with OptiTrack and a NatNet client.
    This code is based on work from Pascal Klink.
    """

    def __init__(self, verbose=False):
        self.info = None
        self.size_delta = 0.5
        self.pos_delta = np.array([0.2, 0.2, 0.2])
        self.lock = Lock()
        self.verbose = verbose

    def create_query(self):
        if self.info is None:
            return {
                "idx": None,
                "pos_bounds": None,
                "size_bounds": None,
                "residual_bounds": None,
            }
        else:
            return {
                "idx": self.info[0],
                "pos_bounds": (
                    self.info[1] - self.pos_delta,
                    self.info[1] + self.pos_delta,
                ),
                "size_bounds": (
                    self.info[2] / (1.0 + self.size_delta),
                    self.info[2] * (1.0 + self.size_delta),
                ),
                "residual_bounds": None,
            }

    def __call__(self, markers):
        self.lock.acquire()

        if len(markers) > 0:
            new_info = markers.find(**(self.create_query()))
            if not isinstance(new_info[0], np.ndarray):
                self.info = new_info
            else:
                n_matches = new_info[0].shape[0]
                if n_matches > 0:
                    self.info = tuple([d[0] for d in new_info])

        if self.verbose:
            print("Cur-Idx: %d" % self.info[0])
            print("Cur-Pos: [%.3e, %.3e, %.3e] " % (self.info[1][0], self.info[1][1], self.info[1][2]))

        self.lock.release()

    def initialized(self):
        self.lock.acquire()
        initialized = self.info is not None
        self.lock.release()
        return initialized

    def get_current_estimate(self):
        self.lock.acquire()
        pos = self.info[1].copy()
        self.lock.release()

        # Change the order of the axis because they are permuted
        return pos[[2, 0, 1]]
