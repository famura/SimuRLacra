import pyrado
from init_args_serializer import Serializable
from pyrado.environments.sim_base import SimEnv
import numpy as np
from pyrado.spaces import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.goalless import OptimProxyTask
from pyrado.tasks.reward_functions import StateBasedRewFcn
from pyrado.utils.data_types import RenderMode
import torch as to
from torch.distributions import MultivariateNormal


class ToyExample(SimEnv, Serializable):
    """
    This environment wraps the 2D toy-example from SNLE.
    One rollout are 4 samples from a multivariate gaussian
    """

    name: str = "ToyExample"

    def __init__(self):
        """ Constructor """
        Serializable._init(self, locals())

        # Initialize basic variables
        super().__init__(dt=None, max_steps=3)

        # Initialize the domain parameters and the derived constants
        self._mean = None
        self._covariance_matrix = None

        self._domain_param = self.get_nominal_domain_param()
        self._calc_constants()

        # Set the bounds for the system's states adn actions
        max_state = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        max_act = np.array([0])
        self._curr_act = np.zeros_like(max_act)  # just for usage in render function

        self._state_space = BoxSpace(-max_state, max_state, labels=["s_1_1", "s_2_1", "s_1_2", "s_2_2", "s_1_3", "s_2_3", "s_1_4", "s_2_4"])
        self._init_space = SingularStateSpace(np.zeros(self._state_space.shape), labels=["s_1_1_init", "s_2_1_init", "s_1_2_init", "s_2_2_init", "s_1_3_init", "s_2_3_init", "s_1_4_init", "s_2_4_init"])
        self._act_space = BoxSpace(-max_act, max_act, labels=["act_1"])
        self._obs_space = None

        # Define the task including the reward function
        self._task = self._create_task()

        # Animation with pyplot
        self._anim = dict(fig=None, trace_x=[], trace_y=[], trace_z=[])

    def _to_scalar(self):
        for param in self._domain_param:
            if isinstance(self._domain_param[param], to.Tensor):    # or isinstance(self._domain_param, np.ndarray):
                self._domain_param[param] = self._domain_param[param].item()

    def _calc_constants(self):
        self._mean, self._covariance_matrix = self.calc_constants(self.domain_param)

    @staticmethod
    def calc_constants(dp):
        for param in dp:
            if isinstance(dp[param], to.Tensor):    # or isinstance(self._domain_param, np.ndarray):
                dp[param] = dp[param].item()
        mean = np.array([dp["m_1"], dp["m_2"]])
        s1 = dp["s_1"] ** 2
        s2 = dp["s_2"] ** 2
        rho = np.tanh(dp["rho"])
        cov12 = rho * s1 * s2
        covariance_matrix = np.array([[s1 ** 2, cov12], [cov12, s2 ** 2]])
        return mean, covariance_matrix

    @property
    def constants(self):
        return self._mean, self._covariance_matrix

    @property
    def state_space(self):
        return self._state_space

    @property
    def obs_space(self):
        return self._state_space

    @property
    def init_space(self):
        return self._init_space

    @property
    def act_space(self):
        return self._act_space

    def _create_task(self, task_args: dict = None) -> OptimProxyTask:
        return OptimProxyTask(self.spec, StateBasedRewFcn(lambda x: 1.00, flip_sign=True))

    @property
    def task(self) -> OptimProxyTask:
        return self._task

    @property
    def domain_param(self):
        return self._domain_param

    @domain_param.setter
    def domain_param(self, param: dict):
        if not isinstance(param, dict):
            raise pyrado.TypeErr(given=param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(param)

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(m_1=0.7,    # first mean
                    m_2=-1.5,    # second mean
                    s_1=-1,   # Sigma_11
                    s_2=-0.9,      # Sigma_22
                    rho=0.6     # scaling factor
                    )  #

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param
            self._calc_constants()

        # Reset the state
        if init_state is None:
            self.step(act=np.array(0))
        else:
            if not init_state.shape == self.obs_space.shape:
                raise pyrado.ShapeErr(given=init_state, expected_match=self.obs_space)
            if isinstance(init_state, np.ndarray):
                self.state = init_state.copy()
            else:
                try:
                    self.state = np.array(init_state)
                except Exception:
                    raise pyrado.TypeErr(given=init_state, expected_type=[np.ndarray, list])

        # Reset time
        self._curr_step = 0

        # No need to reset the task

        # Return perfect observation
        return self.observe(self.state)

    def step(self, act):
        state = np.random.multivariate_normal(self._mean, self._covariance_matrix, size=1).squeeze()

        # Action equal selection a new state a.k.a. solution of the optimization problem
        self.state = state

        # Current reward depending on the state after the step (since there is only one step)
        # self._curr_rew = self.task.step_rew(self.state)
        self._curr_rew = 0

        self._curr_step += 1

        # Check if the task or the environment is done
        done = False
        if self._curr_step == self.max_steps:
            done = True

        return self.observe(self.state), self._curr_rew, done, {}

    @staticmethod
    def log_prob(trajectory, dp: dict):
        """
        Very ugly, but can be used to calculate the probability of a rollout in the case that we are interested on
        the exact posterior probabilty

        Calculates the log-probability for a pair of states and domain parameters.
        """
        mean, covariance_matrix = ToyExample.calc_constants(dp)
        dist = MultivariateNormal(loc=to.tensor(mean), covariance_matrix=to.tensor(covariance_matrix))
        log_prob = to.zeros((1,))
        len_traj = len(trajectory) // 2
        for i in range(len_traj):
            log_prob += dist.log_prob(trajectory[[i, i + 1]])
        return log_prob

    def render(self, mode: RenderMode, render_step: int = 1):
        # Call base class
        super().render(mode)

        # Print to console
        if mode.text:
            if self._curr_step % render_step == 0 and self._curr_step > 0:  # skip the render before the first step
                print(
                    "step: {:3}  |  r_t: {: 1.3f}  |  a_t: {}\t |  s_t+1: {}".format(
                        self._curr_step, self._curr_rew, self._curr_act, self.state
                    )
                )

        # # Render using pyplot
        # if mode.video:
        #     from matplotlib import pyplot as plt
        #     from pyrado.plotting.surface import draw_surface
        #
        #     plt.ion()
        #
        #     if self._anim["fig"] is None:
        #         # Plot Rosenbrock function once if not already plotted
        #         x = np.linspace(-2, 2, 20, True)
        #         y = np.linspace(-1, 3, 20, True)
        #         self._anim["fig"] = draw_surface(x, y, rosenbrock, "x", "y", "z")
        #
        #     self._anim["trace_x"].append(self.state[0])
        #     self._anim["trace_y"].append(self.state[1])
        #     self._anim["trace_z"].append(rosenbrock(self.state))
        #
        #     ax = self._anim["fig"].gca()
        #     ax.scatter(self._anim["trace_x"], self._anim["trace_y"], self._anim["trace_z"], s=8, c="w")
        #
        #     plt.draw()
