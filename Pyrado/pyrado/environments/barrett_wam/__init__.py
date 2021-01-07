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


# 4 DoF arm, 2 DoF actuated
init_qpos_des_4dof = np.array([0.0, 0.6, 0.0, 1.25])
act_min_wam_4dof = np.array([-1.985, -0.9, -10 * np.pi, -10 * np.pi])
act_max_wam_4dof = np.array([1.985, np.pi, 10 * np.pi, 10 * np.pi])
labels_4dof = ["q_1_des", "q_3_des", "q_dot_1_des", "q_dot_3_des"]
act_space_wam_4dof = BoxSpace(act_min_wam_4dof, act_max_wam_4dof, labels=labels_4dof)

# 7 DoF arm, 3 DoF actuated
init_qpos_des_7dof = np.array([0.0, 0.5876, 0.0, 1.36, 0.0, -0.321, -1.57])
act_min_wam_7dof = np.array([-1.985, -0.9, -np.pi / 2, -10 * np.pi, -10 * np.pi, -10 * np.pi])
act_max_wam_7dof = np.array([1.985, np.pi, np.pi / 2, 10 * np.pi, 10 * np.pi, 10 * np.pi])
labels_7dof = ["q_1_des", "q_3_des", "q_5_des", "q_dot_1_des", "q_dot_3_des", "q_dot_5_des"]
act_space_wam_7dof = BoxSpace(act_min_wam_7dof, act_max_wam_7dof, labels=labels_7dof)

# Default PD gains from robcom / SL
wam_pgains = np.array([200.0, 300.0, 100.0, 100.0, 10.0, 10.0, 2.5])
wam_dgains = np.array([7.0, 15.0, 5.0, 2.5, 0.3, 0.3, 0.05])

# Joint limits (with 5 degree safety margin)
qpos_lo = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -2.7]) + 5 * np.pi / 180
qpos_up = np.array([2.6, 1.985, 2.8, 3.14159, 1.25, 1.5707, 2.7]) - 5 * np.pi / 180

# Cartesian cup position when 4dof WAM is in init state
cup_pos_init_sim_4dof = np.array([0.80607, 0.0, 1.4562])
cup_pos_init_sim_7dof = None  # TODO
