import numpy as np
import pathlib

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import aspect2d
from direct.task import Task
from panda3d.core import *

from pyrado.environments.sim_base import SimEnv

# Configuration for panda3d-window
confVars = """
win-size 800 600
framebuffer-multisample 1
multisamples 2
show-frame-rate-meter 1
sync-video 0
threading-model Cull/Cull
"""
loadPrcFileData("", confVars)


class PandaVis(ShowBase):
    def __init__(self):
        """
        Constructor
        """
        ShowBase.__init__(self)
        self.dir = pathlib.Path(__file__).resolve().parent.absolute()

        # Set title and background color
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.windowProperties = WindowProperties()
        self.windowProperties.setForeground(True)
        self.setBackgroundColor(1, 1, 1)

        # Configuration of the lighting
        self.directionalLight1 = DirectionalLight("directionalLight")
        self.directionalLightNP1 = self.render.attachNewNode(
            self.directionalLight1
        )
        self.directionalLightNP1.setHpr(0, -8, 0)
        self.render.setLight(self.directionalLightNP1)

        self.directionalLight2 = DirectionalLight("directionalLight")
        self.directionalLightNP2 = self.render.attachNewNode(
            self.directionalLight2
        )
        self.directionalLightNP2.setHpr(180, -20, 0)
        self.render.setLight(self.directionalLightNP2)

        self.ambientLight = AmbientLight("ambientLight")
        self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
        self.ambientLight.setColor((0.1, 0.1, 0.1, 1))
        self.render.setLight(self.ambientLightNP)

        # Create a text node which displays the parameter on the bottom right of the screen
        self.text = TextNode("parameters")
        self.textNodePath = aspect2d.attachNewNode(self.text)
        self.text.setTextColor(0, 0, 0, 1)
        self.textNodePath.setScale(0.07)

        # Configure trace
        self.trace = LineSegs()
        self.trace.setThickness(3)
        self.trace.setColor(0, 0, 0)
        self.lines = self.render.attachNewNode("Lines")
        self.last_pos = None

    def update(self, task):
        """
        Updates the visualization with every call.

        :param task: Needed by panda3d task manager.
        :return: Task.cont indicates that task should be called again next frame.
        """
        return Task.cont

    def reset(self):
        """
        Resets the the visualization to a certain state, so that in can be run again.
        """
        pass

    def draw_trace(self, point):
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


class QQubeVis(PandaVis):
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of environment class
        self._env = env
        Lr = self._env.domain_param["Lr"]
        Lp = self._env.domain_param["Lp"]
        arm_radius = 0.003
        pole_radius = 0.0045

        # Scaling of the animation so the camera can move smoothly
        self._scale = 20/Lp

        # Set window title
        self.windowProperties.setTitle("Quanser Qube")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setPos(
            -0.4 * self._scale,
            -1.3 * self._scale,
            0.4 * self._scale,
        )
        self.cam.setHpr(-20, -10, 0)

        # Set text properties
        self.textNodePath.setPos(0.4, 0, -0.1)

        # Box
        self.box = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.box.setPos(0, 0.07 * self._scale, 0)
        self.box.setScale(
            0.09 * self._scale,
            0.1 * self._scale,
            0.09 * self._scale,
        )
        self.box.setColor(0.5, 0.5, 0.5)  # grey
        self.box.reparentTo(self.render)

        # Cylinder
        self.cylinder = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_middle.egg")
        )
        self.cylinder.setScale(
            0.005 * self._scale,
            0.005 * self._scale,
            0.03 * self._scale,
        )
        self.cylinder.setPos(0, 0.07 * self._scale, 0.12 * self._scale)
        self.cylinder.setColor(0.5, 0.5, 0, 5)  # grey
        self.cylinder.reparentTo(self.render)

        # Joint 1
        self.joint1 = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.joint1.setScale(0.005 * self._scale)
        self.joint1.setPos(0.0, 0.07 * self._scale, 0.15 * self._scale)
        self.joint1.reparentTo(self.render)

        # Arm
        self.arm = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_top.egg")
        )
        self.arm.setScale(
            arm_radius * self._scale,
            arm_radius * self._scale,
            Lr * self._scale,
        )
        self.arm.setColor(0, 0, 1)  # blue
        self.arm.setP(90)
        self.arm.setPos(0, 0.07 * self._scale, 0.15 * self._scale)
        self.arm.reparentTo(self.render)

        # Joint 2
        self.joint2 = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.joint2.setScale(pole_radius * self._scale)
        self.joint2.setPos(
            0.0,
            (0.07 + 2 * Lr) * self._scale,
            0.15 * self._scale,
        )
        self.joint2.setColor(0, 0, 0)  # black
        self.joint2.wrtReparentTo(self.arm)

        # Pole
        self.pole = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_bottom.egg")
        )
        self.pole.setScale(
            pole_radius * self._scale,
            pole_radius * self._scale,
            Lp * self._scale,
        )
        self.pole.setColor(1, 0, 0)  # red
        self.pole.setPos(0, (0.07 + 2 * Lr) * self._scale, 0.15 * self._scale)
        self.pole.wrtReparentTo(self.arm)

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task: Task):
        # Accessing the current parameter values
        g = self._env.domain_param["g"]
        Mr = self._env.domain_param["Mr"]
        Mp = self._env.domain_param["Mp"]
        Lr = float(self._env.domain_param["Lr"])
        Lp = float(self._env.domain_param["Lp"])
        km = self._env.domain_param["km"]
        Rm = self._env.domain_param["Rm"]
        Dr = self._env.domain_param["Dr"]
        Dp = self._env.domain_param["Dp"]
        th, al, _, _ = self._env.state

        # Update rotation of arm
        self.arm.setH(th * 180 / np.pi - 180)

        # Update rotation of pole
        self.pole.setR(al * 180 / np.pi)

        # Get position of pole
        self.pole_pos = self.pole.getPos(self.render)
        # Calculate position of new point
        self.current_pos = LVecBase3f(
            self.pole_pos[0] + 2 * Lp * np.sin(al) * np.cos(th) * self._scale,
            self.pole_pos[1] + 2 * Lp * np.sin(al) * np.sin(th) * self._scale,
            self.pole_pos[2] - 2 * Lp * np.cos(al) * self._scale)

        # Draw line to that point
        self.draw_trace(self.current_pos)

        # Update displayed text
        self.text.setText(
            f"""
            theta: {self._env.state[0] * 180 / np.pi : 3.1f}
            alpha: {self._env.state[1] * 180 / np.pi : 3.1f}
            dt: {self._env._dt :1.4f}
            g: {g : 1.3f}
            Mr: {Mr : 1.4f}
            Mp: {Mp : 1.4f}
            Lr: {Lr : 1.4f}
            Lp: {Lp : 1.4f}
            Dr: {Dr : 1.7f}
            Dp: {Dp : 1.7f}
            Rm: {Rm : 1.3f}
            km: {km : 1.4f}
            """
        )

        return Task.cont

    def reset(self):
        # Remove the trace
        self.lines.getChildren().detach()
        self.last_pos = None


class PendulumVis(PandaVis):
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize

        """
        super().__init__()

        # Accessing variables of environment class
        self._env = env
        th, _ = self._env.state
        l_pole = float(self._env.domain_param["l_pole"])
        r_pole = 0.05

        # Scaling of the animation so the camera can move smoothly
        self._scale = 10 / l_pole 

        # Set window title
        self.windowProperties.setTitle("Pendulum")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setY(-20 * self._scale)

        # Set text properties
        self.textNodePath.setScale(0.06)
        self.textNodePath.setPos(0.45, 0, -0.3)

        # Joint
        self.joint = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.joint.setPos(0, r_pole * self._scale, 0)
        self.joint.setScale(
            r_pole * self._scale,
            r_pole * self._scale,
            r_pole * self._scale,
        )
        self.joint.setColor(0, 0, 0)  # black
        self.joint.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_top.egg")
        )
        self.pole.setPos(0, r_pole * self._scale, 0)
        self.pole.setScale(
            r_pole * self._scale,
            r_pole * self._scale,
            2 * l_pole * self._scale,
        )
        self.pole.setR(th * 180 / np.pi)
        self.pole.setColor(0, 0, 1)  # blue
        self.pole.reparentTo(self.render)

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task: Task):

        # Accessing the current parameter values
        th, _ = self._env.state
        g = self._env.domain_param["g"]
        m_pole = self._env.domain_param["m_pole"]
        l_pole = float(self._env.domain_param["l_pole"])
        d_pole = self._env.domain_param["d_pole"]
        tau_max = self._env.domain_param["tau_max"]

        # Update position and rotation of pole
        self.pole.setR(th * 180 / np.pi)

        # Update displayed text
        self.text.setText(
            f"""
            dt: {self._env._dt :1.4f}
            theta: {self._env.state[0]*180/np.pi : 2.3f}
            sin theta: {np.sin(self._env.state[0]) : 1.3f}
            cos theta: {np.cos(self._env.state[0]) : 1.3f}
            theta_dot: {self._env.state[1]*180/np.pi : 2.3f}
            tau: {self._env._curr_act[0] : 1.3f}
            g: {g : 1.3f}
            m_pole: {m_pole : 1.3f}
            l_pole: {l_pole : 1.3f}
            d_pole: {d_pole : 1.3f}
            tau_max: {tau_max: 1.3f}
            """
        )

        return Task.cont


class QBallBalancerVis(PandaVis):
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of environment class
        self._env = env
        l_plate = self._env.domain_param["l_plate"]
        m_ball = self._env.domain_param[
            "m_ball"
        ]  # mass of the ball is not needed for panda3d visualization
        r_ball = self._env.domain_param["r_ball"]
        d_plate = 0.01
        r_pole = 0.005
        l_pole = 0.02

        # Scaling of the animation so the camera can move smoothly
        self._scale = 1 / l_plate

        # Set window title
        self.windowProperties.setTitle("Quanser Ball Balancer")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setY(-1.3 * self._scale)

        # Set text properties
        self.textNodePath.setScale(0.05)
        self.textNodePath.setPos(-1.4, 0, 0.9)

        # Ball
        self.ball = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.ball.setPos(
            self._env.state[2] * self._scale,
            self._env.state[3] * self._scale,
            (r_ball + d_plate / 2.0) * self._scale,
        )
        self.ball.setScale(r_ball * self._scale)
        self.ball.setColor(1, 0, 0, 0)  # red
        self.ball.reparentTo(self.render)

        # Plate
        self.plate = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.plate.setScale(
            l_plate * 0.5 * self._scale,
            l_plate * 0.5 * self._scale,
            d_plate * 0.5 * self._scale,
        )  # modified according to Blender object
        self.plate.setColor(0, 1, 1, 0)  # blue
        self.plate.reparentTo(self.render)

        # Joint
        self.joint = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.joint.setPos(0, 0, -d_plate * self._scale)
        self.joint.setScale(
            r_pole * self._scale,
            r_pole * self._scale,
            r_pole * self._scale,
        )
        self.joint.setColor(1, 1, 1)  # white
        self.joint.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_top.egg")
        )
        self.pole.setPos(0, 0, -d_plate * self._scale)
        self.pole.setScale(
            r_pole * self._scale,
            r_pole * self._scale,
            l_pole * self._scale,
        )
        self.pole.setColor(0, 0, 0)  # black
        self.pole.reparentTo(self.render)

        # Null_plate
        self.null_plate = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
        self.null_plate.setPos(0, 0, - 2.5 * l_pole * self._scale)
        self.null_plate.setScale(l_plate * 1.1 * 0.5 * self._scale, l_plate * 1.1 * 0.5 * self._scale, d_plate / 20.0 * self._scale)
        self.null_plate.setTransparency(1)
        self.null_plate.setColorScale(0, 0, 0, 0.5)
        self.null_plate.reparentTo(self.render)

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task: Task):

        # Accessing the current parameter values
        g = self._env.domain_param["g"]
        l_plate = self._env.domain_param["l_plate"]
        m_ball = self._env.domain_param["m_ball"]
        r_ball = self._env.domain_param["r_ball"]
        eta_g = self._env.domain_param["eta_g"]
        eta_m = self._env.domain_param["eta_m"]
        K_g = self._env.domain_param["K_g"]
        J_m = self._env.domain_param["J_m"]
        J_l = self._env.domain_param["J_l"]
        r_arm = self._env.domain_param["r_arm"]
        k_m = self._env.domain_param["k_m"]
        R_m = self._env.domain_param["R_m"]
        B_eq = self._env.domain_param["B_eq"]
        c_frict = self._env.domain_param["c_frict"]
        V_thold_x_neg = self._env.domain_param["V_thold_x_neg"]
        V_thold_x_pos = self._env.domain_param["V_thold_x_pos"]
        V_thold_y_neg = self._env.domain_param["V_thold_y_neg"]
        V_thold_y_pos = self._env.domain_param["V_thold_y_pos"]
        offset_th_x = self._env.domain_param["offset_th_x"]
        offset_th_y = self._env.domain_param["offset_th_y"]
        d_plate = 0.01  # only for animation

        # Get ball position
        x = self._env.state[2]  # along the x axis
        y = self._env.state[3]  # along the y axis

        # Compute plate orientation
        a_vp = -self._env.plate_angs[
            0
        ]  # plate's angle around the y axis (alpha) # Roll
        b_vp = self._env.plate_angs[
            1
        ]  # plate's angle around the x axis (beta) # Pitch

        # Update rotation of plate
        self.plate.setR(-a_vp * 180 / np.pi)  # rotate Roll axis
        self.plate.setP(b_vp * 180 / np.pi)  # rotate Pitch axis

        # Update position of ball
        _current_pos = (
            x * np.cos(a_vp) * self._scale,
            y * np.cos(b_vp) * self._scale,
            (
                r_ball
                + x * np.sin(a_vp)
                + y * np.sin(b_vp)
                + np.cos(a_vp) * d_plate / 2.0
            )
            * self._scale,
        )
        self.ball.setPos(_current_pos)

        # Draw line to that point
        self.draw_trace(_current_pos)

        # Update displayed text
        self.text.setText(
            f"""
            x-axis is pos to the right, y-axis is pos up
            Commanded voltage: x servo : {self._env._curr_act[0] : 1.2f}, y servo : {self._env._curr_act[1] : 1.2f}
            Plate angle around x axis: {self._env.plate_angs[1] * 180 / np.pi : 2.2f}
            Plate angle around y axis: {self._env.plate_angs[0] * 180 / np.pi : 2.2f}
            Shaft angles: {self._env.state[0] * 180 / np.pi : 2.2f}, {self._env.state[1] * 180 / np.pi : 2.2f}
            Ball position: {x : 1.3f}, {y : 1.3f}
            g: {g : 1.3f}
            m_ball: {m_ball : 1.3f}
            r_ball: {r_ball : 1.3f}
            r_arm: {r_arm : 1.3f}
            l_plate: {l_plate : 1.3f}
            K_g: {K_g : 2.2f}
            J_m: {J_m : 1.7f}
            J_l: {J_l : 1.6f}
            eta_g: {eta_g : 1.3f}
            eta_m: {eta_m : 1.3f}
            k_mt: {k_m : 1.3f}
            R_m: {R_m : 1.3f}
            B_eq: {B_eq : 1.3f}
            c_frict: {c_frict : 1.3f}
            V_thold_x_pos: {V_thold_x_pos : 2.3f}
            V_thold_x_neg: {V_thold_x_neg : 2.3f}
            V_thold_y_pos: {V_thold_y_pos : 2.3f}
            V_thold_y_neg: {V_thold_y_neg : 2.3f}
            offset_th_x: {offset_th_x : 2.3f}
            offset_th_y: {offset_th_y : 2.3f}
            """
        )

        return Task.cont

    def reset(self):
        # Remove the trace
        self.lines.getChildren().detach()
        self.last_pos = None


class BallOnBeamVis(PandaVis):
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of environment class
        self._env = env
        r_ball = self._env.domain_param["r_ball"]
        l_beam = self._env.domain_param["l_beam"]
        d_beam = self._env.domain_param["d_beam"]
        x = float(self._env.state[0])  # ball position along the beam axis [m]
        a = float(self._env.state[1])  # angle [rad]

        # Scaling of the animation so the camera can move smoothly
        self._scale = 1 / l_beam

        # Set window title
        self.windowProperties.setTitle("Ball on Beam")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setY(-3.0 * self._scale)

        # Set text properties
        self.textNodePath.setScale(0.07)
        self.textNodePath.setPos(0.3, 0, -0.3)

        # Ball
        self.ball = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.ball.setColor(1, 0, 0, 0)  # red
        self.ball.setScale(r_ball * self._scale)
        self.ball.setPos(
            x * self._scale,
            0,
            (d_beam / 2.0 + r_ball) * self._scale,
        )
        self.ball.reparentTo(self.render)

        # Beam
        self.beam = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.beam.setColor(0, 1, 0, 0)  # green
        self.beam.setScale(
            l_beam / 2 * self._scale,
            d_beam * self._scale,
            d_beam / 2 * self._scale,
        )
        self.beam.setR(-a * 180 / np.pi)
        self.beam.reparentTo(self.render)

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task):

        # Accessing the current parameter values
        g = self._env.domain_param["g"]
        m_ball = self._env.domain_param["m_ball"]
        r_ball = self._env.domain_param["r_ball"]
        m_beam = self._env.domain_param["m_beam"]
        l_beam = self._env.domain_param["l_beam"]
        d_beam = self._env.domain_param["d_beam"]
        ang_offset = self._env.domain_param["ang_offset"]
        c_frict = self._env.domain_param["c_frict"]
        x = float(self._env.state[0])  # ball position along the beam axis [m]
        a = float(self._env.state[1])  # angle [rad]

        # Update position of ball
        self.ball.setPos(
            (np.cos(a) * x - np.sin(a) * (d_beam / 2.0 + r_ball)) * self._scale,
            0,
            (np.sin(a) * x + np.cos(a) * (d_beam / 2.0 + r_ball)) * self._scale,
        )

        # Update rotation of joint
        self.beam.setR(-a * 180 / np.pi)

        # Update displayed text
        self.text.setText(
            f"""
            dt: {self._env._dt : 1.4f}
            g: {g : 1.3f}
            m_ball: {m_ball: 1.2f}
            r_ball: {r_ball : 1.3f}
            m_beam: {m_beam : 1.2f}
            l_beam: {l_beam : 1.2f}
            d_beam: {d_beam : 1.2f}
            c_frict: {c_frict : 1.3f}
            ang_offset: {ang_offset : 1.3f}
            """
        )

        return Task.cont


class QCartPoleVis(PandaVis):
    """
    Visualization for QCartPoleSim
    """
    
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of environment class
        self._env = env
        x, th, _, _ = self._env.state
        l_pole = float(self._env.domain_param["l_pole"])
        l_rail = float(self._env.domain_param["l_rail"])

        # Only for animation
        l_cart, h_cart = 0.08, 0.08
        r_pole, r_rail = 0.01, 0.005

        # Scaling of the animation so the camera can move smoothly
        self._scale = 10 / l_pole

        # Set window title
        self.windowProperties.setTitle("Quanser Cartpole")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setY(-5 * self._scale)

        # Rail
        self.rail = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_middle.egg")
        )
        self.rail.setPos(0, 0, (-h_cart - r_rail) * self._scale)
        self.rail.setScale(
            r_rail * self._scale,
            r_rail * self._scale,
            l_rail * self._scale,
        )
        self.rail.setColor(0.85, 0.85, 0.85)  # light Grey
        self.rail.reparentTo(self.render)
        self.rail.setR(90)

        # Cart
        self.cart = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.cart.setX(x * self._scale)
        self.cart.setScale(
            l_cart * self._scale,
            h_cart / 2 * self._scale,
            h_cart * self._scale,
        )
        self.cart.setColor(0, 1, 0, 0)  # green
        self.cart.reparentTo(self.render)

        # Joint
        self.joint = self.loader.loadModel(
            pathlib.Path(self.dir, "models/ball.egg")
        )
        self.joint.setPos(
            x * self._scale,
            (-r_pole - h_cart / 2) * self._scale,
            0,
        )
        self.joint.setScale(r_pole * self._scale)
        self.joint.setColor(0.85, 0.85, 0.85)  # lightGrey
        self.joint.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(
            pathlib.Path(self.dir, "models/cylinder_center_top.egg")
        )
        self.pole.setPos(
            x * self._scale,
            (-r_pole - h_cart / 2) * self._scale,
            0,
        )
        self.pole.setScale(
            r_pole * self._scale,
            r_pole * self._scale,
            2 * l_pole * self._scale,
        )
        self.pole.setColor(0, 0, 1)  # blue
        self.pole.reparentTo(self.render)

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task):

        # Accessing the current parameter values
        x, th, _, _ = self._env.state
        g = self._env.domain_param["g"]
        m_cart = self._env.domain_param["m_cart"]
        m_pole = self._env.domain_param["m_pole"]
        l_pole = float(self._env.domain_param["l_pole"])
        l_rail = float(self._env.domain_param["l_rail"])
        eta_m = self._env.domain_param["eta_m"]
        eta_g = self._env.domain_param["eta_g"]
        K_g = self._env.domain_param["K_g"]
        J_m = self._env.domain_param["J_m"]
        R_m = self._env.domain_param["R_m"]
        k_m = self._env.domain_param["k_m"]
        r_mp = self._env.domain_param["r_mp"]
        B_eq = self._env.domain_param["B_eq"]
        B_pole = self._env.domain_param["B_pole"]

        # Update position of Cart, Joint and Pole
        self.cart.setX(x * self._scale)
        self.joint.setX(x * self._scale)
        self.pole.setX(x * self._scale)  # could be reparented to cart

        # Update rotation of Pole
        self.pole.setR(-th * 180 / np.pi)

        # Update displayed text
        self.text.setText(
            f"""
            theta: {self._env.state[1] * 180 / np.pi : 2.3f}
            dt: {self._env._dt :1.4f}
            g: {g : 1.3f}
            m_cart: {m_cart : 1.4f}
            l_rail: {l_rail : 1.3f}
            l_pole: {l_pole : 1.3f} (0.168 is short)
            eta_m: {eta_m : 1.3f}
            eta_g: {eta_g : 1.3f}
            K_g: {K_g : 1.3f}
            J_m: {J_m : 1.8f}
            r_mp: {r_mp : 1.4f}
            R_m: {R_m : 1.3f}
            k_m: {k_m : 1.6f}
            B_eq: {B_eq : 1.2f}
            B_pole: {B_pole : 1.3f}
            m_pole: {m_pole : 1.3f}
            """
        )

        return Task.cont

    def reset(self):  # delete?
        pass


class OneMassOscillatorVis(PandaVis):
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of environment class
        self._env = env
        c = 0.1 * self._env.obs_space.bound_up[0]

        # Scaling of the animation so the camera can move smoothly
        self._scale = 5/c 

        # Set window title
        self.windowProperties.setTitle("One Mass Oscilator")
        self.win.requestProperties(self.windowProperties)

        # Set pov
        self.cam.setY(-5 * self._scale)

        # Set text properties
        self.textNodePath.setPos(-1.4, 0, 0.9)

        # Ground
        self.ground = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.ground.setPos(0, 0, -0.02 * self._scale)
        self.ground.setScale(
            self._env.obs_space.bound_up[0] * self._scale,
            1.5 * c * self._scale,
            0.01 * self._scale,
        )  # Scale modified according to Blender Object
        self.ground.setColor(0, 1, 0, 0)  # green
        self.ground.reparentTo(self.render)

        # Object
        self.mass = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.mass.setPos(
            self._env.state[0] * self._scale,
            0,
            c / 2.0 * self._scale,
        )
        self.mass.setScale(
            c * 0.5 * self._scale,
            c * 0.5 * self._scale,
            c * 0.5 * self._scale,
        )  # multiplied by 0.5 since Blender object has length of 2
        self.mass.setColor(0, 0, 1, 0)  # blue
        self.mass.reparentTo(self.render)

        # Desired state
        self.des = self.loader.loadModel(
            pathlib.Path(self.dir, "models/box.egg")
        )
        self.des.setPos(
            self._env._task.state_des[0] * self._scale,
            0,
            0.4 * c * self._scale,
        )
        self.des.setScale(
            0.4 * c * self._scale,
            0.4 * c * self._scale,
            0.4 * c * self._scale,
        )
        self.des.setTransparency(1)
        self.des.setColorScale(0, 1, 1, 0.5)
        self.des.reparentTo(self.render)

        # Force
        self.force = self.loader.loadModel(
            pathlib.Path(self.dir, "models/arrow.egg")
        )
        self.force.setPos(
            self._env.state[0] * self._scale,
            0,
            c / 2.0 * self._scale,
        )
        self.force.setScale(
            0.1 * self._env._curr_act / 10.0 * self._scale,
            0.1 * c * self._scale,
            0.1 * c * self._scale,
        )
        self.force.setColor(1, 0, 0, 0)  # red
        self.force.reparentTo(self.render)

        # Spring
        self.spring = self.loader.loadModel(
            pathlib.Path(self.dir, "models/spring.egg")
        )
        self.spring.setPos(0, 0, c / 2.0 * self._scale)
        self.spring.setScale(
            (self._env.state[0] - c / 2.0) / 7.3 * self._scale,
            c / 6.0 * self._scale,
            c / 6.0 * self._scale,
        )  # scaling according to Blender object
        self.spring.setColor(0, 0, 1, 0)  # blue
        self.spring.reparentTo(self.render)

        # Adds one instance of the update function to the task-manager, thus initializes the animation
        self.taskMgr.add(self.update, "update")

    def update(self, task):

        # Accessing the current parameter values
        m = self._env.domain_param["m"]
        k = self._env.domain_param["k"]
        d = self._env.domain_param["d"]
        c = 0.1 * self._env.obs_space.bound_up[0]

        # Update position of mass and force
        self.mass.setPos(
            self._env.state[0] * self._scale,
            0,
            c / 2.0 * self._scale,
        )
        self.force.setPos(
            self._env.state[0] * self._scale,
            0,
            c / 2.0 * self._scale,
        )

        # Update scale of force
        capped_act = np.sign(self._env._curr_act) * max(0.1 * np.abs(self._env._curr_act), 0.3)
        self.force.setSx(capped_act / 10.0 * self._scale)

        # Update scale of spring
        self.spring.setSx(
            (self._env.state[0] - c / 2.0) / 7.3 * self._scale
        )  # scaling according to Blender object

        # Update displayed text
        self.text.setText(
            f"""
            mass_x: {self.mass.getX()}
            spring_Sx: {self.spring.getSx()}
            dt: {self._env.dt :1.4f}
            m: {m : 1.3f}
            k: {k : 2.2f}
            d: {d : 1.3f}
            """
        )

        return Task.cont

    def reset(self):
        c = 0.1 * self._env.obs_space.bound_up[0]

        self.mass.setPos(self._env.state[0] * self._scale, 0, c / 2.0 * self._scale)
        self.des.setPos(
            self._env._task.state_des[0] * self._scale,
            0,
            0.4 * c * self._scale,
        )
        self.force.setPos(
            self._env.state[0] * self._scale,
            0,
            c / 2.0 * self._scale,
        )
        self.force.setSx((0.1 * self._env._curr_act) / 10.0 * self._scale)
        self.spring.setSx(
            (self._env.state[0] - c / 2.0) / 7.3 * self._scale
        )  # scaling according to Blender object
