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

import os.path as osp
import sys

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import aspect2d
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    DirectionalLight,
    Filename,
    LineSegs,
    NodePath,
    TextNode,
    WindowProperties,
    loadPrcFileData,
)

import pyrado
from pyrado.environments.sim_base import SimEnv


# Configuration for panda3d-window
confVars = """
win-size 1280 720
framebuffer-multisample 1
multisamples 2
show-frame-rate-meter 1
sync-video 0
threading-model Cull/Cull
"""
loadPrcFileData("", confVars)


class PandaVis(ShowBase):
    """Base class for all visualizations with panda3d"""

    def __init__(self, rendering: bool):
        """
        Constructor

        :param rendering: boolean indicating whether to use RenderPipeline or default Panda3d as visualization-module.
        """
        super().__init__(self)
        self.dir = Filename.fromOsSpecific(pyrado.PANDA_ASSETS_DIR).getFullpath()

        # Initialize RenderPipeline
        if rendering:
            sys.path.insert(0, pyrado.RENDER_PIPELINE_DIR)
            from rpcore import RenderPipeline

            self.render_pipeline = RenderPipeline()
            self.render_pipeline.pre_showbase_init()
            self.render_pipeline.set_loading_screen_image(osp.join(self.dir, "logo.png"))
            self.render_pipeline.settings["pipeline.display_debugger"] = False
            self.render_pipeline.create(self)
            self.render_pipeline.daytime_mgr.time = "17:00"
        else:
            self.render_pipeline = None

        # Activate antialiasing
        self.render.setAntialias(AntialiasAttrib.MAuto)

        # Set window properties
        self.windowProperties = WindowProperties()
        self.windowProperties.setForeground(True)
        self.windowProperties.setTitle("Experiment")

        # Set background color
        self.setBackgroundColor(1, 1, 1)

        # Configuration of the lighting
        self.directionalLight1 = DirectionalLight("directionalLight")
        self.directionalLightNP1 = self.render.attachNewNode(self.directionalLight1)
        self.directionalLightNP1.setHpr(0, -8, 0)
        self.render.setLight(self.directionalLightNP1)

        self.directionalLight2 = DirectionalLight("directionalLight")
        self.directionalLightNP2 = self.render.attachNewNode(self.directionalLight2)
        self.directionalLightNP2.setHpr(180, -20, 0)
        self.render.setLight(self.directionalLightNP2)

        self.ambientLight = AmbientLight("ambientLight")
        self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
        self.ambientLight.setColor((0.1, 0.1, 0.1, 1))
        self.render.setLight(self.ambientLightNP)

        # Create a text node displaying the physic parameters on the top left of the screen
        self.text = TextNode("parameters")
        self.textNodePath = aspect2d.attachNewNode(self.text)
        self.text.setTextColor(0, 0, 0, 1)  # black
        self.textNodePath.setScale(0.07)
        self.textNodePath.setPos(-1.9, 0, 0.9)

        # Configure trace
        self.trace = LineSegs()
        self.trace.setThickness(3)
        self.trace.setColor(0.8, 0.8, 0.8)  # light grey
        self.lines = self.render.attachNewNode("Lines")
        self.last_pos = None

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task: Task):
        """
        Updates the visualization with every call.

        :param task: Needed by panda3d task manager.
        :return Task.cont: indicates that task should be called again next frame.
        """
        return Task.cont

    def reset(self):
        """
        Resets the the visualization to a certain state, so that in can be run again. Removes the trace.
        """
        self.lines.getChildren().detach()
        self.last_pos = None

    def draw_trace(self, point):
        """
        Draws a line from the last point to the current point

        :param point: Current position of pen. Needs 3 value vector.
        """
        # Check if trace initialized
        if self.last_pos:
            # Set starting point of new line
            self.trace.moveTo(self.last_pos)

        # Draw line to that point
        self.trace.drawTo(point)

        # Save last position of pen
        self.last_pos = point

        # Show drawing
        self.trace_np = NodePath(self.trace.create())
        self.trace_np.reparentTo(self.lines)


class BallOnBeamVis(PandaVis):
    """Visualisation for the BallOnBeamSim class using panda3d"""

    def __init__(self, env: SimEnv, rendering: bool):
        """
        Constructor

        :param env: environment to visualize
        :param rendering
        """
        super().__init__(rendering)

        # Accessing variables of the environment
        self._env = env
        r_ball = self._env.domain_param["ball_radius"]
        l_beam = self._env.domain_param["beam_length"]
        d_beam = self._env.domain_param["beam_thickness"]
        x = float(self._env.state[0])  # ball position along the beam axis [m]
        a = float(self._env.state[1])  # angle [rad]

        # Scaling of the animation so the camera can move smoothly
        self._scale = 10 / l_beam

        # Set window title
        self.windowProperties.setTitle("Ball on Beam")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setPos(-0.2 * self._scale, -4.0 * self._scale, 0.1 * self._scale)

        # Ball
        self.ball = self.loader.loadModel(osp.join(self.dir, "ball_red.egg"))
        self.ball.setScale(r_ball * self._scale)
        self.ball.setPos(x * self._scale, 0, (d_beam / 2.0 + r_ball) * self._scale)
        self.ball.reparentTo(self.render)

        # Beam
        self.beam = self.loader.loadModel(osp.join(self.dir, "cube_green.egg"))
        self.beam.setScale(l_beam / 2 * self._scale, d_beam * self._scale, d_beam / 2 * self._scale)
        self.beam.setR(-a * 180 / np.pi)
        self.beam.reparentTo(self.render)

    def update(self, task: Task):
        # Accessing the current parameter values
        gravity_const = self._env.domain_param["gravity_const"]
        m_ball = self._env.domain_param["ball_mass"]
        r_ball = self._env.domain_param["ball_radius"]
        m_beam = self._env.domain_param["beam_mass"]
        l_beam = self._env.domain_param["beam_length"]
        d_beam = self._env.domain_param["beam_thickness"]
        c_frict = self._env.domain_param["friction_coeff"]
        ang_offset = self._env.domain_param["ang_offset"]
        x = float(self._env.state[0])  # ball position along the beam axis [m]
        a = float(self._env.state[1])  # angle [rad]

        ball_pos = (
            (np.cos(a) * x - np.sin(a) * (d_beam / 2.0 + r_ball)) * self._scale,
            0,
            (np.sin(a) * x + np.cos(a) * (d_beam / 2.0 + r_ball)) * self._scale,
        )
        # Update position of ball
        self.ball.setPos(ball_pos)

        # Draw trace
        self.draw_trace(ball_pos)

        # Update rotation of joint
        self.beam.setR(-a * 180 / np.pi)

        # Update displayed text
        self.text.setText(
            f"""
            ball position: {self._env.state[0] :1.2f}
            beam angle [deg]: {self._env.state[1] * 180 / np.pi : 3.1f}
            dt: {self._env.dt : 1.4f}
            gravity_const: {gravity_const : 1.3f}
            ball_mass: {m_ball: 1.2f}
            ball_radius: {r_ball : 1.3f}
            beam_mass: {m_beam : 1.2f}
            beam_length: {l_beam : 1.2f}
            beam_thickness: {d_beam : 1.2f}
            friction_coeff: {c_frict : 1.3f}
            ang_offset: {ang_offset : 1.3f}
            """
        )

        return Task.cont


class OneMassOscillatorVis(PandaVis):
    """Visualisation for the OneMassOscillatorSim class using panda3d"""

    def __init__(self, env: SimEnv, rendering: bool):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__(rendering)

        # Accessing variables of the environment
        self._env = env
        c = 0.1 * self._env.obs_space.bound_up[0]

        # Scaling of the animation so the camera can move smoothly
        self._scale = 5 / c

        # Set window title
        self.windowProperties.setTitle("One Mass Oscillator")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setPos(0, -4.0 * self._scale, 0.2 * self._scale)

        # Ground
        self.ground = self.loader.loadModel(osp.join(self.dir, "cube_green.egg"))
        self.ground.setPos(0, 0, -0.02 * self._scale)
        self.ground.setScale(self._env.obs_space.bound_up[0] * self._scale, 1.5 * c * self._scale, 0.01 * self._scale)
        self.ground.reparentTo(self.render)

        # Object
        self.mass = self.loader.loadModel(osp.join(self.dir, "cube_blue.egg"))
        self.mass.setPos(self._env.state[0] * self._scale, 0, c / 2.0 * self._scale)
        self.mass.setScale(c * 0.5 * self._scale, c * 0.5 * self._scale, c * 0.5 * self._scale)
        self.mass.reparentTo(self.render)

        # Desired state
        self.des = self.loader.loadModel(osp.join(self.dir, "cube_green.egg"))
        self.des.setPos(self._env._task.state_des[0] * self._scale, 0, 0.4 * c * self._scale)
        self.des.setScale(0.4 * c * self._scale, 0.4 * c * self._scale, 0.4 * c * self._scale)
        self.des.setTransparency(1)
        self.des.setColorScale(1, 0, 0, 0.5)
        self.des.reparentTo(self.render)

        # Force
        self.force = self.loader.loadModel(osp.join(self.dir, "arrow_red.egg"))
        self.force.setPos(self._env.state[0] * self._scale, 0, c / 2.0 * self._scale)
        self.force.setScale(
            0.1 * self._env._curr_act / 10.0 * self._scale, 0.1 * c * self._scale, 0.1 * c * self._scale
        )
        self.force.reparentTo(self.render)

        # Spring
        self.spring = self.loader.loadModel(osp.join(self.dir, "spring_orange.egg"))
        self.spring.setPos(0, 0, c / 2.0 * self._scale)
        self.spring.setScale(
            (self._env.state[0] - c / 2.0) / 7.3 * self._scale, c / 6.0 * self._scale, c / 6.0 * self._scale
        )
        self.spring.reparentTo(self.render)

    def update(self, task: Task):
        # Accessing the current parameter values
        m = self._env.domain_param["mass"]
        k = self._env.domain_param["stiffness"]
        d = self._env.domain_param["damping"]
        c = 0.1 * self._env.obs_space.bound_up[0]

        # Update position of mass
        pos_mass = (self._env.state[0] * self._scale, 0, c / 2.0 * self._scale)
        self.mass.setPos(pos_mass)
        # And force
        self.force.setPos(self._env.state[0] * self._scale, 0, c / 2.0 * self._scale)

        # Update scale of force
        capped_act = np.sign(self._env._curr_act) * max(0.1 * np.abs(self._env._curr_act), 0.3)
        if capped_act == 0:
            self.force.setSx(0.00001)  # has_mat error if scale = 0
        else:
            self.force.setSx(capped_act / 10.0 * self._scale)

        # Update scale of spring
        self.spring.setSx((self._env.state[0] - c / 2.0) / 7.3 * self._scale)

        # Update displayed text
        self.text.setText(
            f"""
            pos: {self._env.state[0] :1.2f}
            vel: {self._env.state[1] :1.2f}
            dt: {self._env.dt :1.4f}
            mass: {m : 1.3f}
            stiffness: {k : 2.2f}
            damping: {d : 1.3f}
            """
        )

        return Task.cont


class PendulumVis(PandaVis):
    """Visualisation for the PendulumSim class using panda3d"""

    def __init__(self, env: SimEnv, rendering: bool):
        """
        Constructor

        :param env: environment to visualize

        """
        super().__init__(rendering)

        # Accessing variables of the environment
        self._env = env
        th, _ = self._env.state
        l_pole = float(self._env.domain_param["pole_length"])
        r_pole = 0.05

        # Scaling of the animation so the camera can move smoothly
        self._scale = 10 / l_pole

        # Set window title
        self.windowProperties.setTitle("Pendulum")
        self.win.requestProperties(self.windowProperties)

        # Set pov depending on the render mode
        if self.render_pipeline is not None:
            self.cam.setPos(-1 * self._scale, -22 * self._scale, 0)
        else:
            self.cam.setPos(-1 * self._scale, -18 * self._scale, 0)

        # Joint
        self.joint = self.loader.loadModel(osp.join(self.dir, "ball_grey.egg"))
        self.joint.setPos(0, r_pole * self._scale, 0)
        self.joint.setScale(r_pole * self._scale, r_pole * self._scale, r_pole * self._scale)
        self.joint.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(osp.join(self.dir, "cylinder_top_red.egg"))
        self.pole.setPos(0, r_pole * self._scale, 0)
        self.pole.setScale(r_pole * self._scale, r_pole * self._scale, 2 * l_pole * self._scale)
        self.pole.reparentTo(self.render)

    def update(self, task: Task):
        # Accessing the current parameter values
        th, _ = self._env.state
        gravity_const = self._env.domain_param["gravity_const"]
        m_pole = self._env.domain_param["pole_mass"]
        l_pole = float(self._env.domain_param["pole_length"])
        d_pole = self._env.domain_param["pole_damping"]
        tau_max = self._env.domain_param["torque_thold"]

        # Update position and rotation of pole
        self.pole.setR(-th * 180 / np.pi)

        # Get position of pole
        pole_pos = self.pole.getPos(self.render)
        # Calculate position of new point
        current_pos = (
            pole_pos[0] + 4 * l_pole * np.sin(th) * self._scale,
            pole_pos[1],
            pole_pos[2] - 4 * l_pole * np.cos(th) * self._scale,
        )

        # Update displayed text
        self.text.setText(
            f"""
            pole angle [deg]: {self._env.state[0] * 180 / np.pi : 3.1f}
            sin theta: {np.sin(self._env.state[0]) : 1.3f}
            cos theta: {np.cos(self._env.state[0]) : 1.3f}
            theta_dot: {self._env.state[1] * 180 / np.pi : 2.3f}
            torque: {self._env._curr_act[0] : 1.3f}
            dt: {self._env.dt :1.4f}
            gravity_const: {gravity_const : 1.3f}
            pole_mass: {m_pole : 1.3f}
            pole_length: {l_pole : 1.3f}
            pole_damping: {d_pole : 1.3f}
            torque_thold: {tau_max: 1.3f}
            """
        )

        return Task.cont


class QBallBalancerVis(PandaVis):
    """Visualisation for the QBallBalancerSim class using panda3d"""

    def __init__(self, env: SimEnv, rendering: bool):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__(rendering)

        # Accessing variables of the environment
        self._env = env
        l_plate = self._env.domain_param["plate_length"]
        r_ball = self._env.domain_param["ball_radius"]

        # Only for animation
        d_plate = 0.01
        r_pole = 0.005
        l_pole = 0.02

        # Scaling of the animation so the camera can move smoothly
        self._scale = 2 / r_pole

        # Set window title
        self.windowProperties.setTitle("Quanser Ball Balancer")
        self.win.requestProperties(self.windowProperties)

        # Set pov depending on the render mode
        if self.render_pipeline is not None:
            self.cam.setPos(-0.1 * self._scale, -1.0 * self._scale, 0.55 * self._scale)
        else:
            self.cam.setPos(-0.1 * self._scale, -0.8 * self._scale, 0.4 * self._scale)
        self.cam.setHpr(0, -30, 0)  # roll, pitch, yaw [deg]

        # Ball
        self.ball = self.loader.loadModel(osp.join(self.dir, "ball_red.egg"))
        self.ball.setPos(
            self._env.state[2] * self._scale, self._env.state[3] * self._scale, (r_ball + d_plate / 2.0) * self._scale
        )
        self.ball.setScale(r_ball * self._scale)
        self.ball.reparentTo(self.render)

        # Plate
        self.plate = self.loader.loadModel(osp.join(self.dir, "cube_blue.egg"))
        self.plate.setScale(l_plate * 0.5 * self._scale, l_plate * 0.5 * self._scale, d_plate * 0.5 * self._scale)
        self.plate.reparentTo(self.render)

        # Joint
        self.joint = self.loader.loadModel(osp.join(self.dir, "ball_grey.egg"))
        self.joint.setPos(0, 0, -d_plate * self._scale)
        self.joint.setScale(r_pole * self._scale, r_pole * self._scale, r_pole * self._scale)
        self.joint.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(osp.join(self.dir, "cylinder_top_grey.egg"))
        self.pole.setPos(0, 0, -d_plate * self._scale)
        self.pole.setScale(r_pole * self._scale, r_pole * self._scale, l_pole * self._scale)
        self.pole.reparentTo(self.render)

        # Pround plate
        self.null_plate = self.loader.loadModel(osp.join(self.dir, "cube_grey.egg"))
        self.null_plate.setPos(0, 0, -2.5 * l_pole * self._scale)
        self.null_plate.setScale(l_plate * 0.8 * self._scale, l_plate * 0.8 * self._scale, d_plate / 20.0 * self._scale)
        self.null_plate.reparentTo(self.render)

    def update(self, task: Task):
        # Accessing the current parameter values
        gravity_const = self._env.domain_param["gravity_const"]
        l_plate = self._env.domain_param["plate_length"]
        m_ball = self._env.domain_param["ball_mass"]
        r_ball = self._env.domain_param["ball_radius"]
        eta_g = self._env.domain_param["gear_efficiency"]
        eta_m = self._env.domain_param["motor_efficiency"]
        K_g = self._env.domain_param["gear_ratio"]
        J_m = self._env.domain_param["motor_inertia"]
        J_l = self._env.domain_param["load_inertia"]
        r_arm = self._env.domain_param["arm_radius"]
        k_m = self._env.domain_param["motor_back_emf"]
        R_m = self._env.domain_param["motor_resistance"]
        B_eq = self._env.domain_param["combined_damping"]
        ball_damping = self._env.domain_param["ball_damping"]
        V_thold_x_neg = self._env.domain_param["voltage_thold_x_neg"]
        V_thold_x_pos = self._env.domain_param["voltage_thold_x_pos"]
        V_thold_y_neg = self._env.domain_param["voltage_thold_y_neg"]
        V_thold_y_pos = self._env.domain_param["voltage_thold_y_pos"]
        offset_th_x = self._env.domain_param["offset_th_x"]
        offset_th_y = self._env.domain_param["offset_th_y"]
        d_plate = 0.01  # only for animation

        # Get ball position
        x = self._env.state[2]
        y = self._env.state[3]

        # Compute plate orientation
        a_vp = -self._env.plate_angs[0]
        b_vp = self._env.plate_angs[1]

        # Update rotation of plate
        self.plate.setR(-a_vp * 180 / np.pi)
        self.plate.setP(b_vp * 180 / np.pi)

        # Update position of ball
        ball_pos = (
            x * np.cos(a_vp) * self._scale,
            y * np.cos(b_vp) * self._scale,
            (r_ball + x * np.sin(a_vp) + y * np.sin(b_vp) + np.cos(a_vp) * d_plate / 2.0) * self._scale,
        )
        self.ball.setPos(ball_pos)

        # Draw line to that point
        self.draw_trace(ball_pos)

        # Update displayed text
        self.text.setText(  # x-axis is pos to the right, y-axis is pos up
            f"""
            ball pos: {x : 1.3f}, {y : 1.3f}
            plate angle around x axis: {self._env.plate_angs[1] * 180 / np.pi : 2.2f}
            plate angle around y axis: {self._env.plate_angs[0] * 180 / np.pi : 2.2f}
            shaft angles [deg]: {self._env.state[0] * 180 / np.pi : 3.1f}, {self._env.state[1] * 180 / np.pi : 3.1f}
            V_x: {self._env._curr_act[0] : 1.2f}, V_y : {self._env._curr_act[1] : 1.2f}
            gravity_const: {gravity_const : 1.3f}
            ball_mass: {m_ball : 1.3f}
            ball_radius: {r_ball : 1.3f}
            arm_radius: {r_arm : 1.3f}
            plate_length: {l_plate : 1.3f}
            gear_ratio: {K_g : 2.2f}
            motor_inertia: {J_m : 1.7f}
            load_inertia: {J_l : 1.6f}
            gear_efficiency: {eta_g : 1.3f}
            motor_efficiency: {eta_m : 1.3f}
            motor_back_emf: {k_m : 1.3f}
            motor_resistance: {R_m : 1.3f}
            combined_damping: {B_eq : 1.3f}
            ball_damping: {ball_damping : 1.3f}
            V_thold_x_pos: {V_thold_x_pos : 2.3f}
            V_thold_x_neg: {V_thold_x_neg : 2.3f}
            V_thold_y_pos: {V_thold_y_pos : 2.3f}
            V_thold_y_neg: {V_thold_y_neg : 2.3f}
            offset_th_x: {offset_th_x : 2.3f}
            offset_th_y: {offset_th_y : 2.3f}
            """
        )

        return Task.cont


class QCartPoleVis(PandaVis):
    """Visualisation for the QCartPoleSim class using panda3d"""

    def __init__(self, env: SimEnv, rendering: bool):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__(rendering)

        # Accessing variables of the environment
        self._env = env
        x, th, _, _ = self._env.state
        l_pole = float(self._env.domain_param["pole_length"])
        l_rail = float(self._env.domain_param["rail_length"])

        # Only for animation
        l_cart, h_cart = 0.05, 0.045
        r_pole, r_rail = 0.01, 0.005

        # Scaling of the animation so the camera can move smoothly
        self._scale = 10 / l_pole

        # Set window title
        self.windowProperties.setTitle("Quanser Cartpole")
        self.win.requestProperties(self.windowProperties)

        # Set pov depending on the render mode
        if self.render_pipeline is not None:
            self.cam.setPos(-0.2 * self._scale, -3 * self._scale, 0)
        else:
            self.cam.setPos(-0.2 * self._scale, -2 * self._scale, 0)

        # Rail
        self.rail = self.loader.loadModel(osp.join(self.dir, "cylinder_middle_grey.egg"))
        self.rail.setPos(0, 0, (-h_cart - r_rail) * self._scale)
        self.rail.setScale(l_rail / 2 * self._scale, r_rail * self._scale, r_rail * self._scale)
        self.rail.reparentTo(self.render)

        # Cart
        self.cart = self.loader.loadModel(osp.join(self.dir, "cube_green.egg"))
        self.cart.setX(x * self._scale)
        self.cart.setScale(l_cart * self._scale, h_cart / 2 * self._scale, h_cart * self._scale)
        self.cart.reparentTo(self.render)

        # Joint
        self.joint = self.loader.loadModel(osp.join(self.dir, "ball_grey.egg"))
        self.joint.setPos(x * self._scale, (-r_pole - h_cart / 2) * self._scale, 0)
        self.joint.setScale(r_pole * self._scale)
        self.joint.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(osp.join(self.dir, "cylinder_top_red.egg"))
        self.pole.setPos(x * self._scale, (-r_pole - h_cart / 2) * self._scale, 0)
        self.pole.setScale(r_pole * self._scale, r_pole * self._scale, l_pole * self._scale)
        self.pole.reparentTo(self.render)

    def update(self, task: Task):
        # Accessing the current parameter values
        x, th, _, _ = self._env.state
        gravity_const = self._env.domain_param["gravity_const"]
        m_cart = self._env.domain_param["cart_mass"]
        m_pole = self._env.domain_param["pole_mass"]
        l_pole = float(self._env.domain_param["pole_length"])
        l_rail = float(self._env.domain_param["rail_length"])
        eta_m = self._env.domain_param["motor_efficiency"]
        eta_g = self._env.domain_param["gear_efficiency"]
        K_g = self._env.domain_param["gear_ratio"]
        J_m = self._env.domain_param["motor_inertia"]
        R_m = self._env.domain_param["motor_resistance"]
        k_m = self._env.domain_param["motor_back_emf"]
        r_mp = self._env.domain_param["pinion_radius"]
        B_eq = self._env.domain_param["combined_damping"]
        B_pole = self._env.domain_param["pole_damping"]

        # Update position of Cart, Joint and Pole
        self.cart.setX(x * self._scale)
        self.joint.setX(x * self._scale)
        self.pole.setX(x * self._scale)

        # Update rotation of Pole
        self.pole.setR(-th * 180 / np.pi)

        # Get position of pole
        pole_pos = self.pole.getPos(self.render)

        # Draw line to that point
        traced_point = (
            pole_pos[0] + 2 * l_pole * np.sin(th) * self._scale,
            pole_pos[1],
            pole_pos[2] - 2 * l_pole * np.cos(th) * self._scale,
        )
        self.draw_trace(traced_point)

        # Update displayed text
        self.text.setText(
            f"""
            cart position: {x : 1.3f}
            pole angle [deg]: {th * 180 / np.pi : 3.1f}
            dt: {self._env.dt :1.4f}
            gravity_const: {gravity_const : 1.3f}
            cart_mass: {m_cart : 1.4f}
            pole_mass: {m_pole : 1.3f}
            rail_length: {l_rail : 1.3f}
            pole_length: {l_pole : 1.3f}
            motor_efficiency: {eta_m : 1.3f}
            gear_efficiency: {eta_g : 1.3f}
            gear_ratio: {K_g : 1.3f}
            motor_inertia: {J_m : 1.8f}
            pinion_radius: {r_mp : 1.4f}
            motor_resistance: {R_m : 1.3f}
            motor_back_emf: {k_m : 1.6f}
            combined_damping: {B_eq : 1.2f}
            pole_damping: {B_pole : 1.3f}
            """
        )

        return Task.cont


class QQubeVis(PandaVis):
    """Visualisation for the QQubeSim class using panda3d"""

    def __init__(self, env: SimEnv, rendering: bool):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__(rendering)

        # Accessing variables of the environment
        self._env = env
        length_rot_pole = self._env.domain_param["length_rot_pole"]
        length_pend_pole = self._env.domain_param["length_pend_pole"]

        # Only for animation
        arm_radius = 0.0035
        pole_radius = 0.005

        # Scaling of the animation so the camera can move smoothly
        self._scale = 2 / length_pend_pole

        # Set window title
        self.windowProperties.setTitle("Quanser Qube")
        self.win.requestProperties(self.windowProperties)

        # Set pov depending on the render mode
        if self.render_pipeline is not None:
            self.cam.setPos(-0.7 * self._scale, -1.6 * self._scale, 0.3 * self._scale)
            self.cam.setHpr(-20, -5, 0)  # roll, pitch, yaw [deg]
        else:
            self.cam.setPos(-0.6 * self._scale, -1.3 * self._scale, 0.4 * self._scale)
            self.cam.setHpr(-20, -10, 0)  # roll, pitch, yaw [deg]

        # Box
        self.box = self.loader.loadModel(osp.join(self.dir, "cube_green.egg"))
        self.box.setPos(0, 0.07 * self._scale, 0)
        self.box.setScale(0.09 * self._scale, 0.1 * self._scale, 0.09 * self._scale)
        self.box.reparentTo(self.render)

        # Cylinder
        self.cylinder = self.loader.loadModel(osp.join(self.dir, "cylinder_middle_grey.egg"))
        self.cylinder.setScale(0.005 * self._scale, 0.005 * self._scale, 0.03 * self._scale)
        self.cylinder.setPos(0, 0.07 * self._scale, 0.12 * self._scale)
        self.cylinder.reparentTo(self.render)

        # Joint 1
        self.joint1 = self.loader.loadModel(osp.join(self.dir, "ball_grey.egg"))
        self.joint1.setScale(0.005 * self._scale)
        self.joint1.setPos(0.0, 0.07 * self._scale, 0.15 * self._scale)
        self.joint1.reparentTo(self.render)

        # Arm
        self.arm = self.loader.loadModel(osp.join(self.dir, "cylinder_top_blue.egg"))
        self.arm.setScale(arm_radius * self._scale, arm_radius * self._scale, length_rot_pole * self._scale)
        self.arm.setP(90)
        self.arm.setPos(0, 0.07 * self._scale, 0.15 * self._scale)
        self.arm.reparentTo(self.render)

        # Joint 2
        self.joint2 = self.loader.loadModel(osp.join(self.dir, "ball_grey.egg"))
        self.joint2.setScale(pole_radius * self._scale)
        self.joint2.setPos(0.0, (0.07 + 2 * length_rot_pole) * self._scale, 0.15 * self._scale)
        self.joint2.wrtReparentTo(self.arm)

        # Pole
        self.pole = self.loader.loadModel(osp.join(self.dir, "cylinder_bottom_red.egg"))
        self.pole.setScale(pole_radius * self._scale, pole_radius * self._scale, length_pend_pole * self._scale)
        self.pole.setPos(0, (0.07 + 2 * length_rot_pole) * self._scale, 0.15 * self._scale)
        self.pole.wrtReparentTo(self.arm)

    def update(self, task: Task):
        # Accessing the current parameter values
        g = self._env.domain_param["gravity_const"]
        Mr = self._env.domain_param["mass_rot_pole"]
        Mp = self._env.domain_param["mass_pend_pole"]
        Lr = float(self._env.domain_param["length_rot_pole"])
        Lp = float(self._env.domain_param["length_pend_pole"])
        km = self._env.domain_param["motor_back_emf"]
        Rm = self._env.domain_param["motor_resistance"]
        Dr = self._env.domain_param["damping_rot_pole"]
        Dp = self._env.domain_param["damping_pend_pole"]
        th, al, _, _ = self._env.state

        # Update rotation of arm
        self.arm.setH(th * 180 / np.pi - 180)

        # Update rotation of pole
        self.pole.setR(al * 180 / np.pi)

        # Get position of pole
        pole_pos = self.pole.getPos(self.render)

        # Calculate position of new point
        current_pos = (
            pole_pos[0] + 2 * Lp * np.sin(al) * np.cos(th) * self._scale,
            pole_pos[1] + 2 * Lp * np.sin(al) * np.sin(th) * self._scale,
            pole_pos[2] - 2 * Lp * np.cos(al) * self._scale,
        )

        # Draw line to that point
        self.draw_trace(current_pos)

        # Update displayed text
        self.text.setText(
            f"""
            theta [deg]: {self._env.state[0] * 180 / np.pi : 3.1f}
            alpha [deg]: {self._env.state[1] * 180 / np.pi : 3.1f}
            dt: {self._env.dt :1.4f}
            gravity_const: {g : 1.3f}
            mass_rot_pole: {Mr : 1.4f}
            mass_pend_pole: {Mp : 1.4f}
            length_rot_pole: {Lr : 1.4f}
            length_pend_pole: {Lp : 1.4f}
            damping_rot_pole: {Dr : 1.7f}
            damping_pend_pole: {Dp : 1.7f}
            motor_resistance: {Rm : 1.3f}
            motor_back_emf: {km : 1.4f}
            """
        )

        return Task.cont
