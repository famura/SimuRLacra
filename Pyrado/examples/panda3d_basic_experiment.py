from pyrado.environments.pysim.pandavis import PandaVis


"""This file provides a step-by-step example of how to create an Panda3d-animation inside an environment."""

"""
This process requires two rather separate Processes
1.Creating a concrete subclass of PandaVis as a visualization class
2.Implementing animation-specific methods in your environment-class
"""


"""
Part1 Creating a concrete subclass of PandaVis

Start by creating a new class called <<YourEnvironment>>Vis inheriting from PandaVis, inside pandavis.py (Pyrado/pyrado/environments/pysim/pandavis.py)
"""


class YourEnvironmentVis(PandaVis):
    """The following content is extracted from PandaVis.QCartPoleVis"""

    def __init__(self, env, rendering):
        """
        This method will only be called once, at the very start of the simulation. The parameter "env" represent the environment you want to visualize.
        The parameter "rendering" specifies if you want to use RenderPipeline as additional renderer (this will be specified as argument --render).
        """

        """
        Beginn implementing the __init__-method by calling the superconstructor.
        This takes care of most properties shared by all simulations, such as window-Properties, lighting, antialiasing, backgroundcolor, textproperties
        and providing a handy path-variable as well as default version of the trace.
        As mentioned above the "rendering" parameter toggles the usage of the RenderPipeline
        """
        super().__init__(rendering)

        """
        Continue by accessing outer calculated parameters, called environment domain parameters or environment states, by passing them into local variables. 
        Except for the env-variable itself, most of these are specific to the actual simulation and are either entries of the domain_param-dictionary or of the state-dictionary
        The name of the local variable should either be equal to the keyword neccessary to retrive its data from the domain_param-dictionary or be named according to their function (such as x)
        """
        self._env = env
        x, th, _, _ = self._env.state
        l_pole = float(self._env.domain_param["pole_length"])

        """
        Occasionally there is a need for some uncalculated values describing properties irrelevant to the calculations, such as the thickness of a bar, or the radius of a ball
        """
        l_cart, h_cart = 0.08, 0.08
        r_pole, r_rail = 0.01, 0.005

        """
        Now it is neccessary to determine a fixed value as a scaling attribute to enable a handy use of the camera, concerning zooming, rotation and translation.
        This is due to the cameras unchangeable size. It is recommended to choose a value thats dependent on the length-properties of a bigger if not the main primitive object.
        You have to multiply the scaling attribute with every position and scaling of your models (not the rotation!)
        """
        self._scale = 10 / l_pole

        """
        Setting a window title is not absolutely neccessary, but very much recommended and easily done
        """
        self.windowProperties.setTitle("Name of your Environment")
        self.win.requestProperties(self.windowProperties)

        """
        To finish the setup, setting the point of view is neccessary. We recommend a low negative value on the y-axis in order to see the whole animation
        """
        self.cam.setY(-5 * self._scale)

        """
        Optional tweaks can also be implemented now, such as the position or size of the text
        """
        self.textNodePath.setScale(0.05)
        self.textNodePath.setPos(-1.4, 0, 0.9)

        """
        Placing primitive objects. The following indent represents the process for each primitive Object
        """

        """Start by loading the correct model/template for a specific object"""
        self.pole = self.loader.loadModel(osp.join(self.dir, "cylinder_top_blue.egg"))

        """Continuing, there is a number of properties, that can be set, by neatly named accessor methods in most cases, such as
        """
        self.pole.setPos()  # Position(X, Y, Z)
        self.pole.setScale()  # Scale(Sx, Sy, Sz)
        self.pole.setHpr()  # Angles(H, P, R)
        self.pole.setColor()  # Color (R, G, B)
        """
        Position: setPosition(X, Y, Z)
        Scale: setScale(Sx, Sy, Sz)
        Rotation: setHpr(H, P, R)
        Color: setColor(R, G, B, A)    A:alpha/transparency
        """

        """
        At Last, its neccessary to reparent the modified object to the render-instance. This could also be done earlier, with the primitive object being modified afterwards
        """
        self.pole.reparentTo(self.render)

        """
        The last step of the init-method is to add the update-method to the taskmanager, in order for it to call it every frame
        """
        self.taskMgr.add(self.update, "update")

    def update(self, task):
        """
        The update-methode allows objects to move during the animation.
        It is originally called every frame, but with this framerate being dependent on your monitors refreshrate,
        it has been modified in order for the animation to run at the same speed on different monitors
        """

        """
        Similar to the init method, start of by accessing the environments domain parameters and states, being calculated in your environment class
        """
        x, th, _, _ = self._env.state
        l_pole = float(self._env.domain_param["pole_length"])

        """
        Property-updates use the same set of methods as mentioned in the placing-paragraph of the init-method
        """
        # Update position of Cart, Joint and Pole
        self.cart.setX(x * self._scale)
        self.joint.setX(x * self._scale)
        self.pole.setX(x * self._scale)

        # Update rotation of Pole
        self.pole.setR(-th * 180 / np.pi)

        """
        Since every existing simulation required to only update a small amount of properties, we did not specify a spefic order/format for this process.
        However, it is recommended to describe these updates/changes as concrete/specific as possible, by using the single-parameter-methods of the rather abstract accessor-methods,
        such as setX(), setR(), etc. instead of setPos() or setHpr or even setHprPosScale(), as an attempt to make these easily read- and understandable
        """

        """
        Implementing a trace, to visualize the movement-path of an important primitive is a very optional, yet handy feature.
        You only need to calculate the last position of said part of your simulation and pass it into PandaVis.drawTrace(), which is implemented in the superclass
        """
        # Get position of pole
        pole_pos = self.pole.getPos(self.render)
        # Calculate position of new point
        current_pos = (
            pole_pos[0] + 4 * l_pole * np.sin(th) * self._scale,
            pole_pos[1],
            pole_pos[2] - 4 * l_pole * np.cos(th) * self._scale,
        )

        # Draw line to that point
        self.draw_trace(current_pos)

        """
        To illustrate a few important values of your simulation, great for debugging as well ngl, you can print a few of them inside an TextNode-object.
        """
        self.text.setText(
            f"""
                    theta: {self._env.state[1] * 180 / np.pi : 2.3f}
                    dt: {self._env._dt :1.4f}
                    m_cart: {m_cart : 1.4f}
                    l_pole: {l_pole : 1.3f}
                    """
        )

        """
        Finally its neccessary to release the Task, in order to be callable at the next frame
        """
        return Task.cont


"""
Part2: Implementing the _init_anim-method in your environment-class

This is very convenient, as it only consists of two steps.
"""


def _init_anim(self):
    """
    At first it is neccessary to import your Vis-class created in Part1.
    """
    from pyrado.environments.pysim.pandavis import YourEnvironmentVis

    """
    To finally get your simulation started, simply create an instance of said Vis-class
    """
    self._visualization = YourEnvironmentVis(self)
