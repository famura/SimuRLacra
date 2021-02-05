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

        # self.clock = ClockObject.getGlobalClock()
        # self.clock.setMode(ClockObject.M_limited)
        # globalClock.setFrameRate(500)

        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.windowProperties = WindowProperties()
        self.windowProperties.setForeground(True)

        self.directionalLight = DirectionalLight('directionalLight')
        self.directionalLightNP = self.render.attachNewNode(self.directionalLight)
        self.directionalLightNP.setHpr(0, -8, 0)
        self.render.setLight(self.directionalLightNP)

        self.ambientLight = AmbientLight('ambientLight')
        self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
        self.ambientLight.setColor((0.1, 0.1, 0.1, 1))
        self.render.setLight(self.ambientLightNP)

        self.text = TextNode('parameters')
        self.textNodePath = aspect2d.attachNewNode(self.text)

        self.textNodePath.setScale(0.07)


    def update(self, task):
        """
        FrameUpdate

        """
        return Task.cont


class QQubeVis(PandaVis):
    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        self._env = env

        self.windowProperties.setTitle('Quanser Qube')
        self.win.requestProperties(self.windowProperties)

        self.cam.setY(-1.5)
        self.setBackgroundColor(1, 1, 1) #schwarz
        self.textNodePath.setPos(0.4, 0, -0.1)
        self.text.setTextColor(0, 0, 0, 1)

        # Convert to float for VPython
        Lr = float(self._env.domain_param["Lr"])
        Lp = float(self._env.domain_param["Lp"])

        # Init render objects on first call
        scene_range = 0.2
        arm_radius = 0.003
        pole_radius = 0.0045

        self.box = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
        self.box.setPos(0, 0.07, 0)
        self.box.setScale(0.09, 0.1, 0.09)
        self.box.setColor(0.5, 0.5, 0.5)
        self.box.reparentTo(self.render)

        #zeigt nach oben aus Box raus
        self.cylinder = self.loader.loadModel(pathlib.Path(self.dir, "models/cylinder_center_middle.egg"))
        self.cylinder.setScale(0.005, 0.005, 0.03)
        self.cylinder.setPos(0, 0.07, 0.12)
        self.cylinder.setColor(0.5, 0.5, 0,5) #gray
        self.cylinder.reparentTo(self.render)

        # Armself.pole.setPos()
        self.arm = self.loader.loadModel(pathlib.Path(self.dir, "models/cylinder_center_bottom.egg"))
        self.arm.setScale(arm_radius, arm_radius, Lr)
        self.arm.setColor(0, 0, 1) #blue
        self.arm.setP(-90)
        self.arm.setPos(0, 0.07, 0.15)
        self.arm.reparentTo(self.render)

        # Pole
        self.pole = self.loader.loadModel(pathlib.Path(self.dir, "models/cylinder_center_bottom.egg"))
        self.pole.setScale(pole_radius, pole_radius, Lp)
        self.pole.setColor(1, 0, 0) #red
        self.pole.setPos(0, 0.07+2*Lr, 0.15)
        self.pole.wrtReparentTo(self.arm)

        # Joints
        self.joint1 = self.loader.loadModel(pathlib.Path(self.dir, "models/ball.egg"))
        self.joint1.setScale(0.005)
        self.joint1.setPos(0.0, 0.07, 0.15)
        self.joint1.reparentTo(self.render)

        self.joint2 = self.loader.loadModel(pathlib.Path(self.dir, "models/ball.egg"))
        self.joint2.setScale(pole_radius)
        self.joint2.setPos(0.0, 0.07+2*Lr, 0.15)
        self.joint2.setColor(0, 0, 0)
        self.joint2.wrtReparentTo(self.arm)

        self.taskMgr.add(self.update, "update")

    def update(self, task: Task):

        g = self._env.domain_param["g"]
        Mr = self._env.domain_param["Mr"]
        Mp = self._env.domain_param["Mp"]
        Lr = float(self._env.domain_param["Lr"])
        Lp = float(self._env.domain_param["Lp"])
        km = self._env.domain_param["km"]
        Rm = self._env.domain_param["Rm"]
        Dr = self._env.domain_param["Dr"]
        Dp = self._env.domain_param["Dp"]
        #print(globalClock.getDt())
        th, al, _, _ = self._env.state

        self.arm.setH(th*180/np.pi)
        self.pole.setR(-al*180/np.pi)

        # Displayed text
        self.text.setText(f"""
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
            """)

        return Task.cont


class QbbVis(PandaVis):

    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of outer class
        self._env = env

        self.windowProperties.setTitle('Quanser Ball Balancer')
        self.setBackgroundColor(1, 1, 1)
        self.cam.setY(-1.3)

        self.textNodePath.setScale(0.05)
        self.textNodePath.setPos(0.4, 0, -0.1)
        self.text.setTextColor(0, 0, 0, 1)

        # Physics params
        l_plate = self._env.domain_param["l_plate"]
        m_ball = self._env.domain_param["m_ball"]
        r_ball = self._env.domain_param["r_ball"]
        d_plate = 0.01  # only for animation

        # Initiate render objects on first call

        # Ball
        self.ball = self.loader.loadModel(pathlib.Path(self.dir, "models/ball.egg"))
        self.ball.setPos(self._env.state[2], self._env.state[3], (r_ball + d_plate / 2.0))
        self.ball.setScale(r_ball)
        # self.ball.setMass(m_ball)
        self.ball.setColor(1, 0, 0, 0)
        self.ball.reparentTo(self.render)

        # Plate
        self.plate = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
        self.plate.setPos(0, 0, 0)
        self.plate.setScale(l_plate / 2, l_plate / 2, d_plate / 2)
        self.plate.setColor(0, 0, 1, 0)
        self.plate.reparentTo(self.render)

        # Null_plate
        self.null_plate = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
        self.null_plate.setPos(0, 0, 0)
        self.null_plate.setScale(l_plate * 1.1 / 2, l_plate * 1.1 / 2, d_plate / 10 / 2)
        self.null_plate.setTransparency(1)
        self.null_plate.setColorScale(0, 1, 1, 0.5)
        self.null_plate.reparentTo(self.render)

        self.taskMgr.add(self.update, "update")

    def update(self, task: Task):

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

        #  Compute plate orientation
        a_vp = -self._env.plate_angs[0]  # plate's angle around the y axis (alpha) # Roll
        b_vp = self._env.plate_angs[1]  # plate's angle around the x axis (beta) # Pitch

        # Axis runs along the x direction
        self.plate.setScale(l_plate / 2, l_plate / 2, d_plate / 2)

        # self.plate.setHpr(np.cos(a_vp) * 180 / np.pi * float(l_plate), 0, np.sin(a_vp) * 180 / np.pi * float(l_plate))
        self.plate.setR(- a_vp * 180 / np.pi)
        self.plate.setP(b_vp * 180 / np.pi)

        # Get ball position
        x = self._env.state[2]  # along the x axis
        y = self._env.state[3]  # along the y axis

        self.ball.setPos(
            x * np.cos(a_vp),
            y * np.cos(b_vp),
            (r_ball + x * np.sin(a_vp) + y * np.sin(b_vp) + np.cos(a_vp) * d_plate / 2.0),
        )
        self.ball.setScale(r_ball)

        # Set caption text
        self.text.setText(f"""
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
            """)

        return Task.cont


class BobVis(PandaVis):

    def __init__(self, env: SimEnv):
        """
        Constructor

        :param env: environment to visualize
        """
        super().__init__()

        # Accessing variables of outer class
        self._env = env
        r_ball = self._env.domain_param["r_ball"]
        l_beam = self._env.domain_param["l_beam"]
        d_beam = self._env.domain_param["d_beam"]
        x = float(self._env.state[0])  # ball position along the beam axis [m]
        a = float(self._env.state[1])  # angle [rad]

        self.windowProperties.setTitle('Ball on Beam')
        self.cam.setY(-3.0)
        self.textNodePath.setScale(0.07)
        self.textNodePath.setPos(0.3, 0, -0.3)

        self.ball = self.loader.loadModel(pathlib.Path(self.dir, "models/ball.egg"))
        self.ball.setColor(1, 0, 0, 0)
        self.ball.setScale(r_ball)
        self.ball.setPos(x, 0, d_beam / 2.0 + r_ball)
        self.ball.reparentTo(self.render)

        self.beam = self.loader.loadModel(pathlib.Path(self.dir, "models/box.egg"))
        self.beam.setColor(0, 1, 0, 0)
        self.beam.setScale(l_beam / 2, d_beam, d_beam / 2)
        self.beam.setPos(0, 0, 0)
        self.beam.setR(-a * 180 / np.pi)
        self.beam.reparentTo(self.render)

        self.taskMgr.add(self.update, "update")

    def reset(self):
        r_ball = self._env.domain_param["r_ball"]
        d_beam = self._env.domain_param["d_beam"]
        x = float(self._env.state[0])  # ball position along the beam axis [m]
        a = float(self._env.state[1])  # angle [rad]

        self.ball.setPos(x, 0, np.sin(a) * x + np.cos(a) * d_beam / 2.0 + r_ball)

        self.beam.setR(-a * 180 / np.pi)

    def update(self, task):
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

        self.ball.setPos(np.cos(a) * x - np.sin(a) * (d_beam / 2.0 + r_ball), 0,
                         np.sin(a) * x + np.cos(a) * (d_beam / 2.0 + r_ball))

        self.beam.setR(-a * 180 / np.pi)

        # Displayed text
        self.text.setText(f"""
            dt: {self._env._dt : 1.4f}
            g: {g : 1.3f}
            m_ball: {m_ball: 1.2f}
            r_ball: {r_ball : 1.3f}
            m_beam: {m_beam : 1.2f}
            l_beam: {l_beam : 1.2f}
            d_beam: {d_beam : 1.2f}
            c_frict: {c_frict : 1.3f}
            ang_offset: {ang_offset : 1.3f}
            """)

        return Task.cont

