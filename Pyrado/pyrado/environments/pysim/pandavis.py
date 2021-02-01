import pathlib

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import *

# Configuration for panda3d-window
confVars = """
win-size 800 600
framebuffer-multisample 1
multisamples 2
show-frame-rate-meter 1
sync-video #f
vsync #f
"""
loadPrcFileData("", confVars)

class PandaVis(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.dir = pathlib.Path(__file__).resolve().parent.absolute()

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
        return Task.cont
