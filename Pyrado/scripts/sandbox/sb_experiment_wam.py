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
Execute a trajectory on the real WAM using robcom's GoTo command

Dependencies:
    https://git.ias.informatik.tu-darmstadt.de/robcom-2/robcom-2.0

Additional reading:
    Ball-in-a-cup demo:
    https://git.ias.informatik.tu-darmstadt.de/klink/ball-in-a-cup-demo/-/blob/master/bic-new.py
"""

import os.path as osp
import numpy as np
import robcom_python as r

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


def run_direct_control(ex_dir, qpos_des, qvel_des, start_pos):
    def callback(jg, eg, data_provider):
        nonlocal n
        nonlocal time_step
        nonlocal qpos
        nonlocal qvel

        if time_step >= n:
            return True

        dpos = qpos_des[time_step].tolist()
        dvel = qvel_des[time_step].tolist()

        pos = np.array(jg.get(r.JointState.POS))
        vel = np.array(jg.get(r.JointState.VEL))
        qpos.append(pos)
        qvel.append(vel)

        jg.set(r.JointDesState.POS, dpos)
        jg.set(r.JointDesState.VEL, dvel)

        time_step += 1

        return False

    if not len(start_pos) == 7:
        raise pyrado.ShapeErr(given=start_pos, expected_match=np.empty(7))

    # Connect to client
    c = r.Client()
    c.start('192.168.2.2', 2013)  # ip adress and port
    print("Connected to client.")

    # Reset the robot to the initial position
    gt = c.create(r.Goto, "RIGHT_ARM", "")
    gt.add_step(5.0, start_pos)
    print("Moving to initial position")
    gt.start()
    gt.wait_for_completion()
    print("Reached initial position")

    # Read out some states
    group = c.robot.get_group(["RIGHT_ARM"])
    home_qpos = np.array(group.get(r.JointState.POS))
    p_gains = np.array(group.get(r.JointState.P_GAIN))
    d_gains = np.array(group.get(r.JointState.D_GAIN))
    print("Initial (actual) qpos:", home_qpos)
    print("P gain:", p_gains)
    print("D gain:", d_gains)

    input('Hit enter to continue.')

    # Global callback attributes
    n = qpos_des.shape[0]
    time_step = 0
    qpos = []
    qvel = []

    # Start the direct control
    dc = c.create(r.ClosedLoopDirectControl, "RIGHT_ARM", "")
    print("Executing trajectory")
    dc.start(False, 1, callback, ['POS', 'VEL'], [], [])
    dc.wait_for_completion()
    print("Finished execution.")

    print('Measured positions:', np.array(qpos).shape)
    print('Measured velocities:', np.array(qvel).shape)

    np.save(osp.join(ex_dir, 'qpos_real.npy'), qpos)
    np.save(osp.join(ex_dir, 'qvel_real.npy'), qvel)

    c.stop()
    print('Connection closed.')


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Get desired positions and velocities
    if args.mode == 'des':
        # If using the PD controller
        print_cbt('Running desired trajectory ...', 'c', bright=True)
        qpos_exec = np.load(osp.join(ex_dir, 'qpos_des.npy'))
        qvel_exec = np.load(osp.join(ex_dir, 'qvel_des.npy'))
        print_cbt('Saved trajectory into qpos_des.npy and qvel_des.npy', 'g')
    elif args.mode == 'rec':
        # If using WAM's feedforward controller
        print_cbt('Running recorded trajectory ...', 'c', bright=True)
        qpos_exec = np.load(osp.join(ex_dir, 'qpos.npy'))
        qvel_exec = np.load(osp.join(ex_dir, 'qvel.npy'))
        print_cbt('Saved trajectory into qpos.npy and qvel.npy', 'g')
    else:
        raise pyrado.ValueErr(given=args.mode, eq_constraint='des or rec')

    # Run on real WAM
    run_direct_control(ex_dir, qpos_exec, qvel_exec, start_pos=qpos_exec[0, :])
