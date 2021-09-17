import torch as to

import pyrado
from pyrado.algorithms.step_based.a2c import A2C
from pyrado.algorithms.step_based.gae import GAE
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    dt = 1e-3
    env = OneMassOscillatorSim(dt, 5000)

    ex_dir = setup_experiment(OneMassOscillatorSim.name, A2C.name)

    hparam = {
        "particle_hparam": {
            "actor": {"hidden_sizes": [32, 24], "hidden_nonlin": to.relu},
            "vfcn": {"hidden_sizes": [32, 24], "hidden_nonlin": to.relu},
            "critic": {},
        },
        "max_iter": 100,
        "min_steps": 10000,
    }
    particle_param = hparam.pop("particle_hparam")
    actor = FNNPolicy(spec=env.spec, **particle_param["actor"])
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **particle_param["vfcn"])
    critic = GAE(vfcn, **particle_param["critic"])
    algo = A2C(ex_dir, env, actor, critic, **hparam)

    algo.train()
