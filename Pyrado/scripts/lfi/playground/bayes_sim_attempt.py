from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt

from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.special.dummy import IdlePolicy

import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
from delfi.utils.viz import samples_nd

from scripts.lfi.playground.DelfiSimulator import DelfiSimulator
from scripts.lfi.playground.DelfiSummaryStatistics import DelfiSummaryStatistics

if __name__ == "__main__":
    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=200)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake ground truth target domain
    num_real_obs = 1
    env_real = deepcopy(env_sim)
    true_params = dict(k=33, d=0.2)
    env_real.domain_param = true_params
    dp_mapping = {0: "k", 1: "d"}

    # Policy
    behavior_policy = IdlePolicy(env_sim.spec)

    # Create Delfi simulator
    delfi_sim = DelfiSimulator(env_sim, behavior_policy, dp_mapping, "states")

    # define initial prior
    prior_min = np.array([.5, 1e-4])
    prior_max = np.array([80., 1.])
    prior = dd.Uniform(lower=prior_min, upper=prior_max)

    s = DelfiSummaryStatistics()
    g = dg.Default(model=delfi_sim, prior=prior, summary=s)

    # observed data: simulation given true parameters
    obs = delfi_sim.gen_single(list(true_params.values()))

    pilot_samples = 10000

    # training schedule
    n_train = 10000
    n_rounds = 1

    # fitting setup
    minibatch = 10000
    epochs = 100

    # network setup
    n_hiddens = [500, 500]

    # convenience
    prior_norm = True

    # inference object
    obs_stats = s.calc([obs])
    res = infer.SNPEA(g,
                      obs=obs_stats,
                      n_hiddens=n_hiddens,
                      pilot_samples=pilot_samples,
                      prior_norm=prior_norm)

    # train
    log, _, posterior = res.run(
        n_train=n_train,
        n_rounds=n_rounds,
        minibatch=minibatch,
        epochs=epochs)


    print("Generate Plot")
    fig = plt.figure(figsize=(15, 5))
    labels_params = list(true_params.keys())

    # plot loss over iterations
    plt.plot(log[0]["loss"], lw=2)
    plt.xlabel("iteration")
    plt.ylabel("loss")

    prior_min = g.prior.lower
    prior_max = g.prior.upper
    prior_lims = np.concatenate((prior_min.reshape(-1, 1), prior_max.reshape(-1, 1)), axis=1)

    posterior_samples = posterior[0].gen(10000)

    ###################
    # colors
    hex2rgb = lambda h: tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    # RGB colors in [0, 255]
    col = {}
    col["GT"] = hex2rgb("30C05D")
    col["SNPE"] = hex2rgb("2E7FE8")
    col["SAMPLE1"] = hex2rgb("8D62BC")
    col["SAMPLE2"] = hex2rgb("AF99EF")

    # convert to RGB colors in [0, 1]
    for k, v in col.items():
        col[k] = tuple([i / 255 for i in v])

    ###################
    # posterior
    fig, axes = samples_nd(
        posterior_samples,
        limits=prior_lims,
        ticks=prior_lims,
        labels=labels_params,
        fig_size=(5, 5),
        diag="kde",
        upper="kde",
        hist_diag={"bins": 50},
        hist_offdiag={"bins": 50},
        kde_diag={"bins": 50, "color": col["SNPE"]},
        kde_offdiag={"bins": 50},
        points=[np.array(list(true_params.values()))],
        points_offdiag={"markersize": 5},
        points_colors=[col["GT"]],
        title="",
    )

    plt.show()
