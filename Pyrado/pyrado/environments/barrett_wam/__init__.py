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
WAM_Q_LIMITS_LO_7DOF = np.array([-2.6, -2.0, -2.8, -0.9, -4.76, -1.6, -3.0]) + 5 * np.pi / 180
WAM_Q_LIMITS_UP_7DOF = np.array([+2.6, +2.0, +2.8, +3.1, +1.24, +1.6, +3.0]) - 5 * np.pi / 180
# Arbitrarily set velocity limits [rad/s]
WAM_QD_LIMITS_LO_7DOF = -np.array([4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi])
WAM_QD_LIMITS_UP_7DOF = +np.array([4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi, 4 * np.pi])

# Torque limits
# 7 DoF
MAX_TORQUE_7DOF = np.array([150.0, 125.0, 40.0, 60.0, 5.0, 5.0, 2.0])
TORQUE_SPACE_WAM_7DOF = BoxSpace(-MAX_TORQUE_7DOF, MAX_TORQUE_7DOF)
# 4 DoF
MAX_TORQUE_4DOF = MAX_TORQUE_7DOF[:4]
TORQUE_SPACE_WAM_4DOF = BoxSpace(-MAX_TORQUE_4DOF, MAX_TORQUE_4DOF)

# Default PD gains from robcom / SL
WAM_PGAINS_7DOF = np.array([200.0, 300.0, 100.0, 100.0, 10.0, 10.0, 2.5])
WAM_DGAINS_7DOF = np.array([7.0, 15.0, 5.0, 2.5, 0.3, 0.3, 0.05])
WAM_PGAINS_4DOF = WAM_PGAINS_7DOF[:4]
WAM_DGAINS_4DOF = WAM_DGAINS_7DOF[:4]

# Desired initial joint angles
# 7 DoF
INIT_QPOS_DES_7DOF = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])
# 4 DoF arm
INIT_QPOS_DES_4DOF = np.array([0.0, 0.6, 0.0, 1.25])

# Cartesian cup position when WAM is in init state
CUP_POS_INIT_SIM_4DOF = np.array([0.80607, 0.0, 1.4562])  # site name="cup_bottom" (see xml file)
CUP_POS_INIT_SIM_7DOF = None  # not defined yet
GOAL_POS_INIT_SIM_4DOF = np.array([0.8143, 0.0, 1.48])  # site name="cup_goal" (see xml file)
GOAL_POS_INIT_SIM_7DOF = np.array([0.82521, 0.0, 1.4469])

# Action space
# 7 DoF actuated for joint space control
ACT_MIN_JSC_7DOF = np.concatenate([WAM_Q_LIMITS_LO_7DOF, WAM_QD_LIMITS_LO_7DOF])
ACT_MAX_JSC_7DOF = np.concatenate([WAM_Q_LIMITS_UP_7DOF, WAM_QD_LIMITS_UP_7DOF])
labels_jsc_7dof = [f"q_{i}_des" for i in range(1, 8)] + [f"qd_{i}_des" for i in range(1, 8)]
ACT_SPACE_JSC_7DOF = BoxSpace(ACT_MIN_JSC_7DOF, ACT_MAX_JSC_7DOF, labels=labels_jsc_7dof)

# 3 DoF actuated for ball-in-cup task
ACT_MIN_BIC_7DOF = np.array([-1.985, -0.9, -np.pi / 2, -10 * np.pi, -10 * np.pi, -10 * np.pi])
ACT_MAX_BIC_7DOF = np.array([1.985, np.pi, np.pi / 2, 10 * np.pi, 10 * np.pi, 10 * np.pi])
labels_bic_7dof = [f"q_{i}_des" for i in range(1, 6, 2)] + [f"qd_{i}_des" for i in range(1, 6, 2)]
ACT_SPACE_BIC_7DOF = BoxSpace(ACT_MIN_BIC_7DOF, ACT_MAX_BIC_7DOF, labels=labels_bic_7dof)

# Action space
# 4 DoF actuated for pd joint control
ACT_MIN_JSC_4DOF = np.concatenate([WAM_Q_LIMITS_LO_7DOF[:4], WAM_QD_LIMITS_LO_7DOF[:4]])
ACT_MAX_JSC_4DOF = np.concatenate([WAM_Q_LIMITS_UP_7DOF[:4], WAM_QD_LIMITS_UP_7DOF[:4]])
labels_jsc_4dof = [f"q_{i}_des" for i in range(1, 5)] + [f"qd_{i}_des" for i in range(1, 5)]
ACT_SPACE_JSC_4DOF = BoxSpace(ACT_MIN_JSC_4DOF, ACT_MAX_JSC_4DOF, labels=labels_jsc_4dof)

# 2 DoF actuated for ball-in-cup task
ACT_MIN_BIC_4DOF = np.array([-1.985, -0.9, -10 * np.pi, -10 * np.pi])
ACT_MAX_BIC_4DOF = np.array([1.985, np.pi, 10 * np.pi, 10 * np.pi])
labels_bic_4dof = [f"q_{i}_des" for i in range(1, 4, 2)] + [f"qd_{i}_des" for i in range(1, 4, 2)]
ACT_SPACE_BIC_4DOF = BoxSpace(ACT_MIN_BIC_4DOF, ACT_MAX_BIC_4DOF, labels=labels_bic_4dof)
