import torch as to
import numpy as np
import util.math

from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout


class OscillatorTrajectories:
    def __init__(self):
        self.env = OneMassOscillatorSim(dt=0.005, max_steps=200)
        self.policy = IdlePolicy(self.env.spec)

    def __call__(self, mu):
        ro = rollout(self.env, self.policy, eval=True, reset_kwargs=dict(
            domain_param=dict(k=mu[0], d=mu[1])
        ))
        return to.tensor(ro.observations).to(dtype=to.float32).view(-1, 1).squeeze()

