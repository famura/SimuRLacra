import numpy as np
from copy import deepcopy

from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.special.dummy import IdlePolicy

from scripts.lfi.playground.DelfiSimulator import DelfiSimulator

if __name__ == "__main__":

    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=200)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake ground truth target domain
    num_real_obs = 1
    env_real = deepcopy(env_sim)
    env_real.domain_param = dict(k=33, d=0.2)
    dp_mapping = {0: "k", 1: "d"}

    # Policy
    behavior_policy = IdlePolicy(env_sim.spec)

    # Create Delfi simulator
    delfi_sim = DelfiSimulator(env_sim, behavior_policy, dp_mapping, "Ramos")

    delfi_sim.gen_single([33, 0.2])
