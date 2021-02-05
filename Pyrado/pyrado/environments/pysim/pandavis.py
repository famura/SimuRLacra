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


class PendulumVis(PandaVis):
    def __init__(self, env: SimEnv):
        """ 
        Constructor

        :param env: visualization class

        """
        super().__init__()
    
        mydir = pathlib.Path(__file__).resolve().parent.absolute()
    
        # Accessing variables of outer class
        self._env = env
        l_pole = float(self._env.domain_param["l_pole"])
        r_pole = 0.05
        th, _ = self._env.state
        
        # set window title to current Simulation
        self.windowProperties.setTitle('Pendulum')
        self.win.requestProperties(self.windowProperties)
        
        # window and cam properties
        self.setBackgroundColor(1, 1, 1)
        self.cam.setY(-20)
        self.cam.setZ(-0)
        
        # text properties
        self.textNodePath.setColor(0,0,0)
        self.textNodePath.setScale(0.06)
        self.textNodePath.setPos(0.45, 0, -0.3)
        
        # creating the joint object of the pendulum
        self.joint = self.loader.loadModel(pathlib.Path(mydir, "models/ball.egg"))
        self.joint.setPos(0,r_pole,0)
        self.joint.setScale(r_pole,r_pole,r_pole)
        self.joint.setColor(1,1,1)
        self.joint.reparentTo(self.render)
        
        # creating the pole object of the pendulum
        self.pole = self.loader.loadModel(pathlib.Path(mydir, "models/cylinder_center_bottom.egg"))
        self.pole.setPos(0,r_pole,0)
        self.pole.setScale(r_pole,r_pole,2*l_pole)
        self.pole.setR(180*np.pi+180)
        self.pole.setColor(0,0,1)
        self.pole.reparentTo(self.render)
        
        self.taskMgr.add(self.update,"update")
        
    def update(self,task: Task):
        
        # accessing the current parameter values
        g = self._env.domain_param["g"]
        m_pole = self._env.domain_param["m_pole"]
        l_pole = float(self._env.domain_param["l_pole"])
        d_pole = self._env.domain_param["d_pole"]
        tau_max = self._env.domain_param["tau_max"]
        r_pole = 0.05
        th, _ = self._env.state
        
        # change the position and rotation
        self.joint.setPos(0,r_pole,0)
        self.pole.setPos(0,r_pole,0)
        self.pole.setR(th*180/np.pi+180)
        
        # change the shown text
        self.text.setText(f"""
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
            """)
        return Task.cont
    
    def reset(self):
        
        # accessing the current parameter values
        l_pole = float(self._env.domain_param["l_pole"])
        r_pole = 0.05
        th, _ = self._env.state
        
        # change the position and rotation
        self.joint.setPos(0,r_pole,0)
        self.joint.setScale(r_pole,r_pole,r_pole)
        self.joint.setColor(1,1,1)
        
        self.pole.setPos(0,r_pole,0)
        self.pole.setScale(r_pole,r_pole,2*l_pole)
        self.pole.setR(180*np.pi+180)
        self.pole.setColor(0,0,1)
