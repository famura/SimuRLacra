import torch as to

import pyrado
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment


if __name__ == "__main__":
    dt = 1e-3
    env = OneMassOscillatorSim(dt, 5000)

    ex_dir = setup_experiment(OneMassOscillatorSim.name, SVPG.name)

    hparam = {
        "particle_hparam": {
            "actor": {"hidden_sizes": [32, 24], "hidden_nonlin": to.relu},
            "vfcn": {"hidden_sizes": [32, 24], "hidden_nonlin": to.relu},
            "critic": {},
        },
        "max_iter": 100,
        "num_particles": 4,
        "temperature": 0.01,
        "lr": 0.01,
        "horizon": 20,
        "min_steps": 10000,
    }

    algo = SVPG(ex_dir, env, **hparam)

    algo.train()
