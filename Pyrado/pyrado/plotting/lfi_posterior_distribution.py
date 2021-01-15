import inspect

import torch as to
from torch.distributions import MultivariateNormal

from matplotlib import colors
import matplotlib.pyplot as plt

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.utils import BoxUniform

from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer


def plot_posterior_distribution(ax: plt.Axes,
                                posterior: DirectPosterior,
                                observations: to.Tensor,
                                real_environment: DomainRandWrapperBuffer = None,
                                params_names: [str, str] = None,
                                initial_prior: BoxUniform = None,
                                grid_boundaries: [float, float, float, float] = None,
                                color_list = list(colors.TABLEAU_COLORS),
                                grid_size = 1000,
                                scatter_kwargs = dict(),
                                contour_kwargs = dict(),
                                contourf_kwargs = dict()
                                ):
    """
        Create a matplotlib.axes object, that shows the probability distributions of a given posterior wrt. given
        observations as condition.
        Optional: Pass a real_environment object to plot the original environment parameters as a scatter.
        Therefore, the parameter names for the x and y axis have to be given in the params_names object
        (Makes only sense if the real environment is a DomainRandWrapperBuffer object)
        Optional: Pass the initial prior region to set the boundaries of the plotting region. For LFI no
        regions outside the initial prior region can be estimated.
        Optional: Pass the true distribution to plot it as a filled contour in the background (needs a log_prob
        function)

        :param ax: axis of the figure to plot on
        :param posterior: sbi DirectPosterior object
        :param observations: true observed rollouts. Needed as context for the posterior
        :param real_environment: contains the true domain parameters for the given observations
        :param params_names: names of the environmental parameters to plot in x and y direction
        :param initial_prior: initial prior to get the grid boundaries (if grid_boundaries is not given)
        :param grid_boundaries: boundaries of the grid ([x_min, x_max, y_min, y_max])
        :param color_list: list of pyplot colors
        :param grid_size: width and height of the plotting grid
        :param scatter_kwargs: keyword arguments forwarded to pyplot's `scatter()` function for the true parameter
        :param contour_kwargs: keyword arguments forwarded to pyplot's `contour()` function for the posterior distribution
        :param contourf_kwargs: keyword arguments forwarded to pyplot's `contourf()` function for the true distribution
        :return: updated axis ax
        """


    num_obs = observations.shape[0]
    fun_name = inspect.getframeinfo(inspect.currentframe()).function

    # Type Checks
    while len(color_list) < num_obs:
        print("WARNING: Color list is too short. Colors will appear multiple times")
        color_list += color_list

    # define grid using the given grid range
    print("[{}] compute grid ...".format(fun_name))
    if grid_boundaries is not None:
        if type(grid_boundaries) != list or len(grid_boundaries) != 4:
            raise TypeError()
        x = to.linspace(grid_boundaries[0], grid_boundaries[1], grid_size)
        y = to.linspace(grid_boundaries[2], grid_boundaries[3], grid_size)
    # otherwise by using the initial prior boundaries
    elif initial_prior is not None:
        if type(initial_prior) != BoxUniform:
            raise TypeError()
        x = to.linspace(initial_prior.base_dist.low[0],
                        initial_prior.base_dist.high[0],
                        grid_size)
        y = to.linspace(initial_prior.base_dist.low[1],
                        initial_prior.base_dist.high[1],
                        grid_size)
    else:
        raise AttributeError("range or initial_prior has to be given")
    x_mesh, y_mesh = to.meshgrid([x, y])
    pos = to.stack([x_mesh, y_mesh], dim=-1)
    pos = to.flatten(pos, end_dim=-2)
    x_mesh = x_mesh.numpy()
    y_mesh = y_mesh.numpy()

    # plot posteriors
    if type(posterior) != DirectPosterior:
        raise TypeError()
    for obs in range(num_obs):
        print("\r[{}] plot posterior distribution ({}|{}) ...".format(fun_name, obs + 1, num_obs), end="")
        posterior.set_default_x(observations[obs])
        estimated_values = to.exp(posterior.log_prob(pos) - to.max(posterior.log_prob(pos)))
        estimated_values = to.reshape(estimated_values, (grid_size, grid_size)).numpy()
        base_args = dict(colors=color_list[obs], levels=3, zorder=0)  # default
        base_args.update(contour_kwargs)
        plt.contour(x_mesh, y_mesh, estimated_values, **base_args)
    print()

    # plot true distribution
    if real_environment is not None:
        print("[{}] plot real distribution ...".format(fun_name))
        # get the true parameter distribution
        real_distribution = None
        loc = to.tensor([0., 0.])
        covariance_matrix = to.tensor([[0., 0.], [0., 0.]])
        for param in real_environment.randomizer.domain_params:
            for i in range(len(params_names)):
                if param.name == params_names[i]:
                    loc[i] = param.mean
                    covariance_matrix[i, i] = param.std
        try:
            real_distribution = MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
        except Exception as e:
            print("WARNING: Could not build real distribution\n  -- Message:", e)
        # plot
        if real_distribution is not None:
            real_values = to.exp(real_distribution.log_prob(pos))
            real_values = to.reshape(real_values, (grid_size, grid_size)).numpy()
            base_args = dict(zorder=-1)  # default
            base_args.update(contourf_kwargs)
            ax.contourf(x_mesh, y_mesh, real_values, **base_args)

    # plot true observation parameter as scatter
    if type(params_names) is not None and type(real_environment) == DomainRandWrapperBuffer:
        for name in params_names:
            if name not in real_environment.buffer[0].keys():
                raise KeyError("real_environment does not have the parameter {}".format(name))
        print("[{}] plot true parameters ...".format(fun_name))
        true_params = []
        scatter_args = dict()
        for params in real_environment.buffer:
            true_params.append([float(params[params_names[0]]), float(params[params_names[1]])])
        true_params = to.tensor(true_params)

        base_args = dict(c=color_list[:len(true_params)], zorder=1, s=300, marker='X', edgecolors='black')  # default
        base_args.update(scatter_kwargs)
        ax.scatter(true_params[:, 0], true_params[:, 1], **base_args)

    return ax




















