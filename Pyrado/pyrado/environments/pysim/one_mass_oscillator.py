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

import torch.nn as nn
import numpy as np
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.reward_functions import QuadrErrRewFcn

# For OneMassOscillatorDyn
import sys
import torch as to
from typing import Sequence
from tqdm import tqdm
from tabulate import tabulate

from pyrado.sampling.rollout import StepSequence
from pyrado.sampling.utils import gen_batch_idcs


class OneMassOscillatorSim(SimPyEnv, Serializable):
    """  Model of a linear one-mass-oscillator (spring-mass-damper system) without gravity influence """

    name: str = "omo"

    def _create_spaces(self):
        k = self.domain_param["k"]

        # Define the spaces
        max_state = np.array([1.0, 10.0])  # pos [m], vel [m/s]
        min_init_state = np.array([-0.75 * max_state[0], -0.01 * max_state[1]])
        max_init_state = np.array([-0.65 * max_state[0], +0.01 * max_state[1]])
        max_act = np.array([max_state[0] * k])  # max force [N]; should be big enough to reach every steady state
        self._curr_act = np.zeros_like(max_act)  # just for usage in render function

        self._state_space = BoxSpace(-max_state, max_state, labels=["x", "x_dot"])
        self._obs_space = self._state_space
        self._init_space = BoxSpace(min_init_state, max_init_state, labels=["x", "x_dot"])
        self._act_space = BoxSpace(-max_act, max_act, labels=["F"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.zeros(2))
        Q = task_args.get("Q", np.diag([1e1, 1e-2]))
        R = task_args.get("R", np.diag([1e-6]))

        return FinalRewTask(
            DesStateTask(self.spec, state_des, QuadrErrRewFcn(Q, R)),
            factor=1e3,
            mode=FinalRewMode(always_negative=True),
        )

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            m=1.0, k=30.0, d=0.5  # object's mass [kg]  # spring stiffness constant [N/m]
        )  # damping constant [Ns/m]

    def _calc_constants(self):
        m = self.domain_param["m"]
        k = self.domain_param["k"]
        d = self.domain_param["d"]

        self.omega = np.sqrt(k / m)  # eigen frequency [Hz]
        self.zeta = d / (2.0 * np.sqrt(m * k))  # damping ratio [-]
        if self.zeta < 1.0:
            self._omega_d = np.sqrt(1 - self.zeta ** 2) * self.omega  # damped eigen frequency [Hz]
        else:
            self._omega_d = None  # overdamped case, no oscillation
        if self.zeta < np.sqrt(1 / 2):
            self._omega_res = np.sqrt(1 - 2.0 * self.zeta ** 2) * self.omega  # resonance frequency [Hz]
        else:
            self._omega_res = None  # damping too high, no resonance

    def _step_dynamics(self, act: np.ndarray):
        m = self.domain_param["m"]

        # Linear Dynamics
        A = np.array([[0, 1], [-self.omega ** 2, -2.0 * self.zeta * self.omega]])
        B = np.array([[0], [1.0 / m]])
        state_dot = A.dot(self.state) + B.dot(act).reshape(2)

        # Integration Step (forward Euler)
        self.state = self.state + state_dot * self._dt  # next state

    def _init_anim(self):
        from pyrado.environments.pysim.pandavis import PandaVis
        from direct.task import Task
        import pathlib

        class PandaVisOmo(PandaVis):

            def __init__(self, omo):
                super().__init__()

                # Accessing variables of outer class
                self.omo = omo

                self.setBackgroundColor(0, 0, 0)
                self.cam.setY(-5)
                self.cam.setZ(1)
                self.cam.setP(-10)
                self.textNodePath.setPos(0.4, 0, -0.1)
                self.text.setTextColor(1, 1, 1, 1)

                # Params
                c = 0.1 * self.omo.obs_space.bound_up[0]

                # Ground
                #ToDo Mass RoM and Ground_size do not appeal
                self.ground = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
                self.ground.setPos(0, 0, -0.02)
                self.ground.setScale(
                    2 * self.omo.obs_space.bound_up[0],
                    3 * c,
                    0.02
                )
                self.ground.setColor(0, 1, 0, 0)
                self.ground.reparentTo(self.render)

                # Mass
                self.mass = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
                self.mass.setPos(self.omo.state[0], 0, c / 2.0)
                self.mass.setScale(c, c, c)
                self.mass.setColor(0, 0, 1, 0)
                self.mass.reparentTo(self.render)

                # Des
                self.des = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
                self.des.setPos(self.omo._task.state_des[0], 0, 0.8 * c / 2.0)
                self.des.setScale(0.8 * c, 0.8 * c, 0.8 * c)
                self.des.setTransparency(1)
                self.des.setColorScale(0, 1, 1, 0.5)
                self.des.reparentTo(self.render)

                # Force
                self.force = self.loader.loadModel(pathlib.Path(self.dir, "models/arrow.egg"))
                self.force.setPos(self.omo.state[0], 0, c / 2.0)
                #self.force.setScale(0.1 * self.omo._curr_act, 0.2 * c, 0.2 * c)
                self.force.setScale(0.2, 0.2 * c, 0.2 * c)
                self.force.setColor(1, 0, 0, 0)
                self.force.reparentTo(self.render)

                # Spring
                self.spring = self.loader.loadModel(pathlib.Path(self.dir, "models/spring.egg"))
                self.spring.setPos(0, 0, c / 2.0)
                self.spring.setScale(self.omo.state[0] - c / 2.0, c / 3.0, c / 3.0)
                #self.spring.setScale(0.1, 0.1, 0.1)
                self.spring.setColor(0, 0, 1, 0)
                self.spring.reparentTo(self.render)

                self.taskMgr.add(self.update, "update")

            def update(self, task):

                m = self.omo.domain_param["m"]
                k = self.omo.domain_param["k"]
                d = self.omo.domain_param["d"]
                c = 0.1 * self.omo.obs_space.bound_up[0]

                #ToDo Mass moves too little
                self.mass.setPos(self.omo.state[0], 0, c / 2.0)

                #ToDo Force does not change at runtime
                #weil capped_act ändert sich nicht
                #eventuell nicht richtig gesetzt in der Sandbox, dummyWert = 0
                #sollte sich da nicht eigentlich die Scale ändern?
                self.force.setPos(self.mass.getPos())
                capped_act = np.sign(self.omo._curr_act) * max(0.1 * np.abs(self.omo._curr_act), 0.3)
                #self.force.setHpr(capped_act, 0, 0)

                #ToDo Spring moves too much
                #sollte anhand mass ausgerichtet werden, siehe nächstes T0D0
                #self.spring.setSx(self.omo.state[0] - c / 2.0)
                self.spring.setSx(self.omo.state[0] / 10)

                #ToDo Mass_center and Spring_end do not align
                #weil spring.setSx() mit Faktoren arbeitet und mass.setPos() mit Koordinaten
                #brauchen eine Funktion f, für die gilt spring.setSx(f(mass.getPos())), sodass mass_center = spring_end

                # set caption text
                self.text.setText(f"""
                    mass_x: {self.mass.getX()}
                    spring_Sx: {self.spring.getSx()}
                    dt: {self.omo.dt :1.4f}
                    m: {m : 1.3f}
                    k: {k : 2.2f}
                    d: {d : 1.3f}
                    """)

                return Task.cont

            def reset(self):
                c = 0.1 * self.omo.obs_space.bound_up[0]

                self.mass.setPos(self.omo.state[0], 0, c / 2.0)
                self.des.setPos(self.omo._task.state_des[0], 0, 0.8 * c / 2.0)
                self.force.setPos(self.omo.state[0], 0, c / 2.0)
                self.force.setHpr(0.1 * self.omo._curr_act, 0, 0)
                self.spring.setSx(self.omo.state[0] - c / 2.0)

        # Create instance of PandaVis
        self._visualization = PandaVisOmo(self)
        # States that visualization is running
        self._initiated = True

    def _update_anim(self):
        # Refreshed with every frame
        self._visualization.taskMgr.step()

    def _reset_anim(self):
        self._visualization.reset()


class OneMassOscillatorDyn(Serializable):
    def __init__(self, dt: float):
        """
        Constructor

        :param dt: simulation step size [s]
        """
        Serializable._init(self, locals())

        self._dt = dt
        self.omega = None
        self.zeta = None
        self.A = None
        self.B = None

    def _calc_constants(self, domain_param: dict):
        self.omega = to.sqrt(domain_param["k"] / domain_param["m"])
        self.zeta = domain_param["d"] / (2.0 * to.sqrt(domain_param["m"] * domain_param["k"]))

        self.A = to.stack([to.tensor([0.0, 1.0]), to.stack([-self.omega ** 2, -2.0 * self.zeta * self.omega])])
        self.B = to.stack([to.tensor(0.0), 1.0 / domain_param["m"]]).view(-1, 1)

    def __call__(self, state: to.Tensor, act: to.Tensor, domain_param: dict) -> to.Tensor:
        """
        One step of the forward dynamics

        :param state: current state
        :param act: current action
        :param domain_param: current domain parameters
        :return: next state
        """
        self._calc_constants(domain_param)
        # act = self.limit_act(act)

        # state_dot = self.A @ state + self.B @ act
        state_dot = state @ self.A.t() + act @ self.B.t()  # Pyro batching

        # Return the state delta (1st order approximation)
        return state_dot * self._dt


class OneMassOscillatorDomainParamEstimator(nn.Module):
    """ Class to estimate the domain parameters of the OneMassOscillator environment """

    def __init__(self, dt: float, dp_init: dict, num_epoch: int, batch_size: int):
        super().__init__()

        self.dp_est = nn.Parameter(to.tensor([dp_init["m"], dp_init["k"], dp_init["d"]]), requires_grad=True)
        self.dp_fixed = dict(dt=dt)

        self.optim = to.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        self.loss_fcn = nn.MSELoss()
        self.num_epoch = num_epoch
        self.batch_size = batch_size

        # Define the dynamics
        self.dyn = OneMassOscillatorDyn(dt)

    def forward(self, state: to.Tensor, act: to.Tensor) -> to.Tensor:
        return self.dyn(state, act, dict(m=self.dp_est[0], k=self.dp_est[1], d=self.dp_est[2]))

    def update(self, rollouts: Sequence[StepSequence]):
        # Pre-process rollout data
        [ro.torch(data_type=to.get_default_dtype()) for ro in rollouts]
        states_cat = to.cat([ro.observations[:-1] for ro in rollouts])
        actions_cat = to.cat([ro.actions for ro in rollouts])
        targets_cat = to.cat([(ro.observations[1:] - ro.observations[:-1]) for ro in rollouts])  # state deltas

        # Iteration over the full data set
        for e in range(self.num_epoch):
            loss_list = []

            # Mini-batch optimization
            for idcs in tqdm(
                gen_batch_idcs(self.batch_size, len(targets_cat)),
                total=(len(targets_cat) + self.batch_size - 1) // self.batch_size,
                desc=f"Epoch {e}",
                unit="batches",
                file=sys.stdout,
                leave=False,
            ):
                # Make predictions
                preds = to.stack([self.forward(s, a) for s, a in zip(states_cat[idcs], actions_cat[idcs])])

                # Reset the gradients and call the optimizer
                self.optim.zero_grad()
                loss = self.loss_fcn(preds, targets_cat[idcs])
                loss.backward()
                loss_list.append(loss.detach().cpu().numpy())
                self.optim.step()

            print(tabulate([["avg loss", np.mean(loss_list)], ["param estimate", self.dp_est.detach().cpu().numpy()]]))
