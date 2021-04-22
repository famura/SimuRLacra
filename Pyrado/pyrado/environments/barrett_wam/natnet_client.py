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

import logging
import socket
import struct
from threading import Thread
from typing import Callable, NoReturn, Optional

import numpy as np


# Create structs for reading various object types to speed up parsing.
Vector3 = struct.Struct("<fff")
Quaternion = struct.Struct("<ffff")
IntValue = struct.Struct("<i")
FloatValue = struct.Struct("<f")
DoubleValue = struct.Struct("<d")


class LabeledMarkers:
    """
    Helper class for keeping track of markers.
    This code is based on work from Pascal Klink.
    """

    def __init__(self):
        self.idx_dict = {}
        self.idcs = []
        self.positions = []
        self.sizes = []
        self.residuals = []

    def __len__(self):
        return len(self.idx_dict)

    def add(self, idx, pos, size, res):
        self.idcs.append(idx)
        self.positions.append(pos)
        self.sizes.append(size)
        self.residuals.append(size)
        self.idx_dict[idx] = len(self.positions) - 1

    def finalize(self):
        self.idcs = np.array(self.idcs)
        self.positions = np.array(self.positions)
        self.sizes = np.array(self.sizes)
        self.residuals = np.array(self.residuals)

    def find(self, idx=None, pos_bounds=None, size_bounds=None, residual_bounds=None):
        if idx is not None and idx in self.idx_dict:
            des_id = self.idx_dict[idx]
            return (
                self.idcs[des_id],
                self.positions[des_id, :],
                self.sizes[des_id],
                self.residuals[des_id],
            )
        else:
            if pos_bounds is not None:
                in_pos_bounds = np.logical_and(
                    np.all(self.positions >= pos_bounds[0][None, :], axis=1),
                    np.all(self.positions <= pos_bounds[1][None, :], axis=1),
                )
            else:
                in_pos_bounds = np.ones_like(self.sizes, dtype=bool)

            if size_bounds is not None:
                in_size_bounds = np.logical_and(self.sizes >= size_bounds[0], self.sizes <= size_bounds[1])
            else:
                in_size_bounds = np.ones_like(self.sizes, dtype=bool)

            if residual_bounds is not None:
                in_residual_bounds = np.logical_and(
                    self.residuals >= residual_bounds[0],
                    self.residuals <= residual_bounds[1],
                )
            else:
                in_residual_bounds = np.ones_like(self.sizes, dtype=bool)

            in_bounds = np.logical_and(in_pos_bounds, in_residual_bounds, in_size_bounds)
            return (
                self.idcs[in_bounds],
                self.positions[in_bounds, :],
                self.sizes[in_bounds],
                self.residuals[in_bounds],
            )

    def get_ids(self):
        return self.idx_dict.keys()


class NatNetClient:
    """
    Client for communicating with OptiTrack.
    This code is based on work from Boris Belousov and Pascal Klink.
    """

    rigidBodyListener: Optional[Callable[[int, str, Vector3, Quaternion], NoReturn]]
    labeled_marker_listener: Optional[Callable[[LabeledMarkers], NoReturn]]
    newFrameListener: Optional[Callable[[int, int, int, int, int, int, int, int, DoubleValue, bool, bool], NoReturn]]

    def __init__(
        self,
        ver=(3, 0, 0, 0),
        server_ip="192.168.2.3",
        gateway_ip="192.168.2.1",
        quiet=True,
    ):
        """
        :param ver: version as tuple
        :param server_ip: IP address of the NatNet server, where Motive is running
        :param gateway_ip: IP address of the gateway. Set to None to read out this IP from socket (which was not
                           working in our tests)
        :param quiet: Set to False to print additional NatNet messages using coloredlogs lib
        """
        self.__natNetStreamVersion = ver
        self.serverIPAddress = server_ip
        self.gateway_ip = gateway_ip
        self.multicastAddress = "239.255.42.99"
        self.commandPort = 1510
        self.dataPort = 1511
        self.rb_map = {}  # mapping from IDs to rigid body names
        self.dataSocket = None
        self.commandSocket = None

        # Callbacks
        self.rigidBodyListener = None
        self.newFrameListener = None
        self._stop = False
        self.labeled_marker_listener = None

        # Logging
        if not quiet:
            import coloredlogs  # pylint: disable=wrong-import-order

            coloredlogs.install(
                level="INFO",
                fmt="NatNet: %(message)s",
                level_styles={"info": {"color": "black"}},
            )

    # Client/server message ids
    NAT_PING = 0
    NAT_PINGRESPONSE = 1
    NAT_REQUEST = 2
    NAT_RESPONSE = 3
    NAT_REQUEST_MODELDEF = 4
    NAT_MODELDEF = 5
    NAT_REQUEST_FRAMEOFDATA = 6
    NAT_FRAMEOFDATA = 7
    NAT_MESSAGESTRING = 8
    NAT_DISCONNECT = 9
    NAT_UNRECOGNIZED_REQUEST = 100

    # ============================= Data unpacking ============================ #
    def __unpackMarkerSet(self, data):
        offset = 0
        # Model name
        modelName, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(modelName) + 1
        logging.info("\tModel Name: {}".format(modelName.decode("utf-8")))
        # Marker count
        markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tMarker Count: {}".format(markerCount))
        # Markers
        for j in range(markerCount):
            pos = Vector3.unpack(data[offset : offset + 12])
            offset += 12
            logging.info("\t\tMarker {}: {}".format(j, pos))
        return offset

    def __unpackRigidBody(self, data):
        offset = 0

        # ID (4 bytes)
        rb_id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tID: {}".format(rb_id))

        # Position and orientation
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        logging.info("\t\tPosition: {}".format(pos))
        rot = Quaternion.unpack(data[offset : offset + 16])
        offset += 16
        logging.info("\t\tOrientation: {}".format(rot))

        # After Version 3.0, marker data is in description
        ver = self.__natNetStreamVersion
        if ver[0] < 3:
            # Marker count (4 bytes)
            markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            logging.info("\t\tMarker Count: {}".format(markerCount))
            markerCountRange = range(markerCount)

            # Marker positions
            for i in markerCountRange:
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                logging.info("\t\t\tMarker {}: {}".format(i, pos))

            if ver[0] >= 2:
                # Marker ID's
                for i in markerCountRange:
                    marker_id = int.from_bytes(data[offset : offset + 4], byteorder="little")
                    offset += 4
                    logging.info("\t\t\tMarker {} ID {}".format(i, marker_id))

                # Marker sizes
                for i in markerCountRange:
                    size = FloatValue.unpack(data[offset : offset + 4])
                    offset += 4
                    logging.info("\t\t\tMarker {} Size {}".format(i, size))

        # NatNet version 2.0 and later
        if ver[0] >= 2:
            (markerError,) = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            logging.info("\t\tMean marker error: {}".format(markerError))

        # Version 2.6 and later
        if ver[0] == 2 and ver[1] >= 6 or ver[0] > 2:
            (param,) = struct.unpack("h", data[offset : offset + 2])
            offset += 2
            trackingValid = (param & 0x01) != 0
            logging.info("\t\tTracking Valid: {}".format(trackingValid))

        # Send information to any listener.
        if self.rigidBodyListener is not None:
            if rb_id in self.rb_map:
                self.rigidBodyListener(rb_id, self.rb_map[rb_id], pos, rot)  # pylint: disable=not-callable

        return offset

    def __unpackSkeleton(self, data):
        offset = 0
        # Skeleton ID
        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tID: {}".format(id))
        # Rigid body count
        rigidBodyCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tRigid Body Count: {}".format(rigidBodyCount))
        # Rigid bodies
        for j in range(rigidBodyCount):
            offset += self.__unpackRigidBody(data[offset:])

        return offset

    def __unpackLabeledMarker(self, data):
        offset = 0
        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        model_id = id >> 16
        marker_id = id & 0x0000FFFF
        logging.info("\tModelID: {}, MarkerID: {}".format(model_id, marker_id))
        pos = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        logging.info("\t\tPosition: {}".format(pos))
        (size,) = FloatValue.unpack(data[offset : offset + 4])
        offset += 4
        logging.info("\t\tSize: {}".format(size))

        # Version 2.6 and later
        ver = self.__natNetStreamVersion
        if ver[0] == 2 and ver[1] >= 6 or ver[0] > 2:
            (param,) = struct.unpack("h", data[offset : offset + 2])
            offset += 2
            occluded = (param & 0x01) != 0
            pointCloudSolved = (param & 0x02) != 0
            modelSolved = (param & 0x04) != 0
            if ver[0] >= 3:
                hasModel = (param & 0x04) != 0
                unlabeled = (param & 0x10) != 0
                activeMarker = (param & 0x20) != 0

        # Version 3.0 and later
        if ver[0] >= 3:
            (residual,) = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            logging.info("\t\tResidual: {}".format(residual))

        return offset, marker_id, pos, size, residual

    def __unpackForcePlate(self, data):
        offset = 0
        # Force plate ID
        forcePlateID = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tForce Plate {}".format(forcePlateID))
        # Channel Count
        forcePlateChannelCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        # Channel Data
        for j in range(forcePlateChannelCount):
            logging.info("\t\tChannel {}".format(j))
            forcePlateChannelFrameCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            for k in range(forcePlateChannelFrameCount):
                forcePlateChannelVal = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                logging.info("\t\t\t {}".format(forcePlateChannelVal))
        return offset

    def __unpackDevice(self, data):
        offset = 0
        # ID
        deviceID = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tDevice {}".format(deviceID))
        # Channel Count
        deviceChannelCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        # Channel Data
        for j in range(deviceChannelCount):
            logging.info("\t\tChannel {}".format(j))
            deviceChannelFrameCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            for k in range(deviceChannelFrameCount):
                deviceChannelVal = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                logging.info("\t\t\t {}".format(deviceChannelVal))
        return offset

    def __unpackMocapData(self, data):
        data = memoryview(data)
        offset = 0

        # Frame number (4 bytes)
        frameNumber = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("Frame: {}".format(frameNumber))

        # ================ Marker sets
        markerSetCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("Marker Set Count: {}".format(markerSetCount))
        for i in range(markerSetCount):
            offset += self.__unpackMarkerSet(data[offset:])

        # ================ Unlabeled markers (DEPRECATED)
        ver = self.__natNetStreamVersion
        unlabeledMarkersCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        if ver[0] < 3:
            logging.info("Unlabeled Marker Count: {}".format(unlabeledMarkersCount))
            for i in range(unlabeledMarkersCount):
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                logging.info("\tMarker {} : {}".format(i, pos))
        else:
            # Just skip them
            offset += 12 * unlabeledMarkersCount

        # ================ Rigid bodies
        rigidBodyCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("Rigid Body Count: {}".format(rigidBodyCount))
        for i in range(rigidBodyCount):
            offset += self.__unpackRigidBody(data[offset:])

        # ================ Skeletons (Version 2.1 and later)
        skeletonCount = 0
        if ver[0] == 2 and ver[1] > 0 or ver[0] > 2:
            skeletonCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            logging.info("Skeleton Count: {}".format(skeletonCount))
            for i in range(skeletonCount):
                offset += self.__unpackSkeleton(data[offset:])

        # ================ Labeled markers (Version 2.3 and later)
        labeledMarkerCount = 0
        if ver[0] == 2 and ver[1] >= 3 or ver[0] > 2:
            labeledMarkerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            logging.info("Labeled Marker Count: {}".format(labeledMarkerCount))
            markers = LabeledMarkers()
            for i in range(labeledMarkerCount):
                tmp = self.__unpackLabeledMarker(data[offset:])
                offset += tmp[0]
                markers.add(*tmp[1:])

            markers.finalize()
            if self.labeled_marker_listener is not None:
                self.labeled_marker_listener(markers)  # pylint: disable=not-callable

        # ================ Force Plate data (version 2.9 and later)
        if ver[0] == 2 and ver[1] >= 9 or ver[0] > 2:
            forcePlateCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            logging.info("Force Plate Count: {}".format(forcePlateCount))
            for i in range(forcePlateCount):
                offset += self.__unpackForcePlate(data[offset:])

        # ================ Device data (version 2.11 and later)
        if ver[0] == 2 and ver[1] >= 11 or ver[0] > 2:
            deviceCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            logging.info("Device Count: {}".format(deviceCount))
            for i in range(deviceCount):
                offset += self.__unpackDevice(data[offset:])

        # ================ Timing
        if ver[0] < 3:
            softwareLatency = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
            logging.info("Software latency: {}".format(softwareLatency))

        # Timecode
        timecode = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        timecodeSub = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        # Timestamp (increased to double precision in 2.7 and later)
        if ver[0] == 2 and ver[1] >= 7 or ver[0] > 2:
            (timestamp,) = DoubleValue.unpack(data[offset : offset + 8])
            offset += 8
        else:
            (timestamp,) = FloatValue.unpack(data[offset : offset + 4])
            offset += 4
        logging.info("Timestamp: {}".format(timestamp))

        # Hires Timestamp (Version 3.0 and later)
        if ver[0] >= 3:
            stampCameraExposure = int.from_bytes(data[offset : offset + 8], byteorder="little")
            offset += 8
            logging.info("Mid-exposure timestamp: {}".format(stampCameraExposure))
            stampDataReceived = int.from_bytes(data[offset : offset + 8], byteorder="little")
            offset += 8
            logging.info("Camera data received timestamp: {}".format(stampDataReceived))
            stampTransmit = int.from_bytes(data[offset : offset + 8], byteorder="little")
            offset += 8
            logging.info("Transmit timestamp: {}".format(stampTransmit))

        # ================ Frame parameters
        trackedModelsChanged = False
        isRecording = trackedModelsChanged
        if ver[0] >= 3:
            (param,) = struct.unpack("h", data[offset : offset + 2])
            offset += 2
            isRecording = (param & 0x01) != 0
            trackedModelsChanged = (param & 0x02) != 0

        # Send information to any listener
        if self.newFrameListener is not None:
            self.newFrameListener(  # pylint: disable=not-callable
                frameNumber,
                markerSetCount,
                unlabeledMarkersCount,
                rigidBodyCount,
                skeletonCount,
                labeledMarkerCount,
                timecode,
                timecodeSub,
                timestamp,
                isRecording,
                trackedModelsChanged,
            )

    # ======================= Data description unpacking ====================== #
    def __unpackMarkerSetDescription(self, data):
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        logging.info("\tMarkerset Name: {}".format(name.decode("utf-8")))

        markerCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tMarker Count: {}".format(markerCount))

        for i in range(markerCount):
            name, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(name) + 1
            logging.info("\t\tMarker Name: {}".format(name.decode("utf-8")))

        return offset

    def __unpackRigidBodyDescription(self, data):
        offset = 0

        # Rigid body name (NatNet 2.0 and later)
        if self.__natNetStreamVersion[0] >= 2:
            name, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(name) + 1
            logging.info("\tRigid Body Name: {}".format(name.decode("utf-8")))

        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tID: {}".format(id))
        self.rb_map[id] = name

        parentID = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tParent ID: {}".format(parentID))

        coord_offset = Vector3.unpack(data[offset : offset + 12])
        offset += 12
        logging.info("\tOffset: {}".format(coord_offset))

        # Per-marker data (NatNet 3.0 and later)
        if self.__natNetStreamVersion[0] >= 3:
            n_markers = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4

            for marker_idx in range(n_markers):
                pos = Vector3.unpack(data[offset : offset + 12])
                offset += 12
                logging.info("\t\tMarker {}: {}".format(marker_idx, pos))

            for marker_idx in range(n_markers):
                marker_required_label = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                if marker_required_label != 0:
                    logging.info("\t\tRequired active label: {}".format(marker_required_label))

        return offset

    def __unpackSkeletonDescription(self, data):
        offset = 0

        name, separator, remainder = bytes(data[offset:]).partition(b"\0")
        offset += len(name) + 1
        logging.info("\tSkeleton Name: {}".format(name.decode("utf-8")))

        id = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tID: {}".format(id))

        if id in self.rb_map and self.rb_map[id] != name:
            raise RuntimeError(
                "Rigid body '%s' is already present in the current map but with name '%s'!" % (name, self.rb_map[id])
            )

        rigidBodyCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("\tRigidBody (Bone) Count: {}".format(rigidBodyCount))

        for i in range(rigidBodyCount):
            offset += self.__unpackRigidBodyDescription(data[offset:])

        return offset

    def __unpackDataDescriptions(self, data):
        offset = 0
        datasetCount = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        logging.info("Dataset Count: {}".format(datasetCount))

        # Clear the rigid body ID map
        self.rb_map = {}
        for i in range(datasetCount):
            logging.info("\tDataset: {}".format(i))
            type = int.from_bytes(data[offset : offset + 4], byteorder="little")
            offset += 4
            logging.info("\tType: {}".format(type))
            if type == 0:
                offset += self.__unpackMarkerSetDescription(data[offset:])
            elif type == 1:
                offset += self.__unpackRigidBodyDescription(data[offset:])
            elif type == 2:
                offset += self.__unpackSkeletonDescription(data[offset:])

    # ================================ Threads ================================ #
    def __processMessage(self, data):
        logging.info("\n------------\nBegin Packet")

        messageID = int.from_bytes(data[0:2], byteorder="little")
        logging.info("Message ID: {}".format(messageID))

        packetSize = int.from_bytes(data[2:4], byteorder="little")
        logging.info("Packet Size: {}".format(packetSize))

        offset = 4
        if messageID == self.NAT_FRAMEOFDATA:
            self.__unpackMocapData(data[offset:])
        elif messageID == self.NAT_MODELDEF:
            self.__unpackDataDescriptions(data[offset:])
        elif messageID == self.NAT_PINGRESPONSE:
            name, _, _ = bytes(data[offset:]).partition(b"\0")
            logging.info("\tApp Name: {}".format(name.decode("utf-8")))
            offset += 256  # Skip the sending app's Name field
            logging.info("\tApp Version: {}".format(struct.unpack("BBBB", data[offset : offset + 4])))
            offset += 4  # Skip the sending app's Version info
            self.__natNetStreamVersion = struct.unpack("BBBB", data[offset : offset + 4])
            logging.info("\tApp NatNet Version: {}".format(self.__natNetStreamVersion))
            offset += 4
        elif messageID == self.NAT_RESPONSE:
            if packetSize == 4:
                commandResponse = int.from_bytes(data[offset : offset + 4], byteorder="little")
                offset += 4
                logging.info("Command response: {}".format(commandResponse))
            else:
                message, separator, remainder = bytes(data[offset:]).partition(b"\0")
                offset += len(message) + 1
                logging.info("Command response: {}".format(message.decode("utf-8")))
        elif messageID == self.NAT_UNRECOGNIZED_REQUEST:
            logging.info("Received 'Unrecognized request' from server")
        elif messageID == self.NAT_MESSAGESTRING:
            message, separator, remainder = bytes(data[offset:]).partition(b"\0")
            offset += len(message) + 1
            logging.info("Received message from server: {}".format(message.decode("utf-8")))
        else:
            logging.info("ERROR: Unrecognized packet type")

        logging.info("End Packet\n----------")

    def __threadFunction(self, socket):
        while not self._stop:
            # Block for input
            data, addr = socket.recvfrom(32768)  # 32k byte buffer size
            if len(data) > 0:
                self.__processMessage(data)

    # ================================ Sockets ================================ #
    def __createDataSocket(self, port):
        result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)  # Internet  # UDP
        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if self.gateway_ip is None:
            # A hacky way to get the IP of the interface that connects to the Internet
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.gateway_ip = s.getsockname()[0]
            s.close()

        # Specify my_ip as the interface to subscribe to multicast through
        result.setsockopt(
            socket.SOL_IP,
            socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(self.multicastAddress) + socket.inet_aton(self.gateway_ip),
        )

        # # Original way of signing up for multicast
        # mreq = struct.pack("4sl", socket.inet_aton(self.multicastAddress),
        #                    socket.INADDR_ANY)
        # result.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        result.bind((self.multicastAddress, port))
        return result

    def __createCommandSocket(self):
        result = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        result.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        result.bind(("", 0))
        result.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        return result

    # ================================= Main ================================== #
    def sendCommand(self, command, commandStr, socket, address):
        packetSize = 0
        if command == self.NAT_REQUEST:
            packetSize = len(commandStr) + 1

        data = command.to_bytes(2, byteorder="little")
        data += packetSize.to_bytes(2, byteorder="little")

        data += commandStr.encode("utf-8")
        data += b"\0"

        socket.sendto(data, address)

    def run(self, daemon=True):
        """
        Start the data and command threads. The attached callback functions (listener) should now retrieve data.

        :param daemon: If set to True, the data and command thread are closed as soon as the main thread has finished.
                       Otherwise, the threads remain open and the program has to be shut down manually.
        """
        # Data socket and thread
        self.dataSocket = self.__createDataSocket(self.dataPort)
        if self.dataSocket is None:
            raise RuntimeError("Could not open data channel")
        dataThread = Thread(target=self.__threadFunction, args=(self.dataSocket,))
        dataThread.daemon = daemon

        # Command socket and thread
        self.commandSocket = self.__createCommandSocket()
        if self.commandSocket is None:
            raise RuntimeError("Could not open command channel")
        commandThread = Thread(target=self.__threadFunction, args=(self.commandSocket,))
        commandThread.daemon = daemon

        # # Request server name, version, and NatNet version
        commandThread.start()
        # self.sendCommand(self.NAT_PING, "", self.commandSocket,
        #                  (self.serverIPAddress, self.commandPort))
        # time.sleep(0.1)
        # # Request frame data definition
        self.sendCommand(
            self.NAT_REQUEST_MODELDEF,
            "",
            self.commandSocket,
            (self.serverIPAddress, self.commandPort),
        )
        # time.sleep(0.1)
        dataThread.start()

    def stop(self):
        """ Stops receiving data from socket """
        self._stop = True
