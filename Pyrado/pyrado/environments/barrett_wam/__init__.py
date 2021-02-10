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

from pyrado.spaces import BoxSpace


# 7 DoF joint limits [rad] (with 5 degree safety margin)
# See https://support.barrett.com/wiki/WAM/KinematicsJointRangesConversionFactors
wam_q_limits_lo_7dof = np.array([-2.6, -2.0, -2.8, -0.9, -4.76, -1.6, -3.0]) + 5 * np.pi / 180
wam_q_limits_up_7dof = np.array([+2.6, +2.0, +2.8, +3.1, +1.24, +1.6, +3.0]) - 5 * np.pi / 180
# Arbitrarily set velocity limits [rad/s]
wam_qd_limits_lo_7dof = -np.array([4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])
wam_qd_limits_up_7dof = +np.array([4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])

# Torque limits
# 7 DoF
max_torque_7dof = np.array([150.0, 125.0, 40.0, 60.0, 5.0, 5.0, 2.0])
torque_space_wam_7dof = BoxSpace(-max_torque_7dof, max_torque_7dof)
# 4 DoF
max_torque_4dof = max_torque_7dof[:4]
torque_space_wam_4dof = BoxSpace(-max_torque_4dof, max_torque_4dof)

# Default PD gains from robcom / SL
wam_pgains_7dof = np.array([200.0, 300.0, 100.0, 100.0, 10.0, 10.0, 2.5])
wam_dgains_7dof = np.array([7.0, 15.0, 5.0, 2.5, 0.3, 0.3, 0.05])
wam_pgains_4dof = wam_pgains_7dof[:4]
wam_dgains_4dof = wam_dgains_7dof[:4]

# Desired initial joint angles
# 7 DoF
init_qpos_des_7dof = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])
# 4 DoF arm
init_qpos_des_4dof = np.array([0.0, 0.6, 0.0, 1.25])

# Cartesian cup position when WAM is in init state
cup_pos_init_sim_4dof = np.array([0.80607, 0.0, 1.4562])  # site name="cup_bottom" (see xml file)
cup_pos_init_sim_7dof = None  # TODO
goal_pos_init_sim_4dof = np.array([0.8143, 0.0, 1.48])  # site name="cup_goal" (see xml file)
goal_pos_init_sim_7dof = np.array([0.82521, 0.0, 1.4469])

# Action space
# 7 DoF actuated for joint space control
act_min_jsc_7dof = np.concatenate([wam_q_limits_lo_7dof, wam_qd_limits_lo_7dof])
act_max_jsc_7dof = np.concatenate([wam_q_limits_up_7dof, wam_qd_limits_up_7dof])
labels_jsc_7dof = [f"q_{i}_des" for i in range(1, 8)] + [f"q_dot_{i}_des" for i in range(1, 8)]
act_space_jsc_7dof = BoxSpace(act_min_jsc_7dof, act_max_jsc_7dof, labels=labels_jsc_7dof)

# 3 DoF actuated for ball-in-cup task
act_min_bic_7dof = np.array([-1.985, -0.9, -np.pi / 2, -10 * np.pi, -10 * np.pi, -10 * np.pi])
act_max_bic_7dof = np.array([1.985, np.pi, np.pi / 2, 10 * np.pi, 10 * np.pi, 10 * np.pi])
labels_bic_7dof = [f"q_{i}_des" for i in range(1, 6, 2)] + [f"q_dot_{i}_des" for i in range(1, 6, 2)]
act_space_bic_7dof = BoxSpace(act_min_bic_7dof, act_max_bic_7dof, labels=labels_bic_7dof)

# Action space
# 4 DoF actuated for pd joint control
act_min_jsc_4dof = np.concatenate([wam_q_limits_lo_7dof[:4], wam_qd_limits_lo_7dof[:4]])
act_max_jsc_4dof = np.concatenate([wam_q_limits_up_7dof[:4], wam_qd_limits_up_7dof[:4]])
labels_jsc_4dof = [f"q_{i}_des" for i in range(1, 5)] + [f"q_dot_{i}_des" for i in range(1, 5)]
act_space_jsc_4dof = BoxSpace(act_min_jsc_4dof, act_max_jsc_4dof, labels=labels_jsc_4dof)

# 2 DoF actuated for ball-in-cup task
act_min_bic_4dof = np.array([-1.985, -0.9, -10 * np.pi, -10 * np.pi])
act_max_bic_4dof = np.array([1.985, np.pi, 10 * np.pi, 10 * np.pi])
labels_bic_4dof = [f"q_{i}_des" for i in range(1, 4, 2)] + [f"q_dot_{i}_des" for i in range(1, 4, 2)]
act_space_bic_4dof = BoxSpace(act_min_bic_4dof, act_max_bic_4dof, labels=labels_bic_4dof)
