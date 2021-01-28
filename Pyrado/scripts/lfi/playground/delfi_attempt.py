import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from delfi.simulator.BaseSimulator import BaseSimulator
import delfi.distribution as dd
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats
import delfi.generator as dg
import delfi.inference as infer
from delfi.utils.viz import samples_nd
import torch as torch


class BasicSim(BaseSimulator):
    def __init__(self):
        dim_param = 2
        super().__init__(dim_param=dim_param, seed=None)
        self.sim = sim

    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = np.asarray(params)
        assert params.ndim == 1, "params.ndim must be 1"
        state = self.sim(params)
        return {"data": state.reshape(-1)}


class BasicSum(BaseSummaryStats):
    def __init__(self, seed=None):
        super(BasicSum, self).__init__(seed=seed)

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        return np.array([repetition_list[0].get("data")])


def sim(parameter_set):
    foo = np.array([])
    for i in range(len(parameter_set)):
        foo = np.append(foo, parameter_set[i] + np.random.normal(0, 0.1))
    return foo


num_dim = 2
pilot_samples = 2000

# training schedule
n_train = 2000
n_rounds = 3

# fitting setup
minibatch = 100
epochs = 100

seed_p = 2
prior_min = np.array([-1, -1])
prior_max = np.array([1, 1])
prior = dd.Uniform(lower=prior_min, upper=prior_max, seed=seed_p)

# define model, prior, summary statistics and generator classes
parameter_set = np.array([1, 1])
m = BasicSim()
s = BasicSum()
g = dg.Default(model=m, prior=prior, summary=s)

# true parameters and respective labels

labels_params = [r"$p_{1}$", r"$p_{2}$"]
true_params = np.array([0.0, 0.0])
# observed data: simulation given true parameters
obs = m.gen_single(true_params)
obs_stats = s.calc([obs])
print(obs_stats)

# inference object
res = infer.SNPEA(
    generator=g,
    obs=obs_stats,
)

# train
log, _, posterior = res.run(
    n_train=n_train,
    n_rounds=n_rounds,
    minibatch=minibatch,
    epochs=epochs,
)


fig = plt.figure(figsize=(15, 5))

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
    points=[true_params],
    points_offdiag={"markersize": 5},
    points_colors=[col["GT"]],
    title="",
)

plt.show()
