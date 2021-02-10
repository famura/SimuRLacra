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
Run a PD-controller with the parameter from Quanser on the real device.
By default all controllers in this script run infinitely.
"""
import torch as to
from datetime import datetime

import pyrado
from pyrado.environments.quanser.quanser_ball_balancer import QBallBalancerReal
from pyrado.environments.quanser.quanser_cartpole import QCartPoleSwingUpReal, QCartPoleStabReal
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.policies.special.environment_specific import (
    QBallBalancerPDCtrl,
    QCartPoleSwingUpAndBalanceCtrl,
    QQubeSwingUpAndBalanceCtrl,
)
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    if args.env_name in QBallBalancerReal.name:
        env = QBallBalancerReal(args.dt, args.max_steps)
        policy = QBallBalancerPDCtrl(env.spec, kp=to.diag(to.tensor([3.45, 3.45])), kd=to.diag(to.tensor([2.11, 2.11])))
        print_cbt("Set up controller for the QBallBalancerReal environment.", "c")

    elif args.env_name == QCartPoleStabReal.name:
        env = QCartPoleStabReal(args.dt, args.max_steps)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        print_cbt("Set up controller for the QCartPoleStabReal environment.", "c")

    elif args.env_name == QCartPoleSwingUpReal.name:
        env = QCartPoleSwingUpReal(args.dt, args.max_steps)
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        print_cbt("Set up controller for the QCartPoleSwingUpReal environment.", "c")

    elif args.env_name == QQubeReal.name:
        env = QQubeReal(args.dt, args.max_steps)
        policy = QQubeSwingUpAndBalanceCtrl(env.spec)
        print_cbt("Set up controller for the QQubeReal environment.", "c")

    else:
        raise pyrado.ValueErr(
            given=args.env_name,
            eq_constraint=f"{QBallBalancerReal.name}, {QCartPoleSwingUpReal.name}, "
            f"{QCartPoleStabReal.name}, or {QQubeReal.name}",
        )

    # Run on device
    done = False
    print_cbt("Running predefined controller ...", "c", bright=True)
    while not done:
        ro = rollout(env, policy, eval=True, render_mode=RenderMode(text=args.verbose))

        if args.save_figures:
            pyrado.save(
                ro,
                "rollout_real",
                "pkl",
                pyrado.TEMP_DIR,
                meta_info=dict(suffix=datetime.now().strftime(pyrado.timestamp_format)),
            )
            print_cbt(f"Saved rollout to {pyrado.TEMP_DIR}", "g")

        done, _, _ = after_rollout_query(env, policy, ro)
