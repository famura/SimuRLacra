import torch as to
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_2d_thetas(proposals, obs_thetas=None, marginal_samples=None):
    """
    proposals: to.Tensor[num_observations, num_samples, parameter_size]
    obs_thetas: to.Tensor[num_observations, theta_parameter_size]
    """
    if proposals.shape[2] != 2:
        raise TypeError("Theta has to be two dimensional")

    # max min and range values for x and y
    x_max = to.max(proposals[:, :, 0])
    x_min = to.min(proposals[:, :, 0])

    y_max = to.max(proposals[:, :, 1])
    y_min = to.min(proposals[:, :, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min

    # create figure
    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS)  # list of color keys
    legend_elements = [Line2D([0], [0], marker='o', color='black', label='Sampled Thetas', markersize=10)]

    # plot samples from proposals
    cnt = 0
    for obs in proposals:
        ax.scatter(obs[:, 0], obs[:, 1],
                   c=colors[cnt] ,label="Samples p(theta|x_o_{}".format(cnt + 1), alpha=0.3)
        cnt += 1

    # plot marginal samples
    if marginal_samples is not None:
        ax.scatter(marginal_samples[:, 0], marginal_samples[:, 1], c=colors[-1],
                   label="Marg. Samples", alpha=0.3)

    # plot original observation thetas
    if obs_thetas is not None:
        legend_elements.append(Line2D([0], [0], marker='x', color='black', label='Original Theta', markersize=10))
        cnt = 0
        for obs in obs_thetas:
            ax.scatter(obs[0], obs[1], facecolors=colors[cnt],
                       marker='X', edgecolors='black', s=200)
            cnt += 1

    # set axis limits (offset on the edges by 'factor' * 100%)
    factor = 0.1
    ax.set_xlim((x_min - x_range * factor, x_max + x_range * factor))
    ax.set_ylim((y_min - y_range * factor, y_max + y_range * factor))

    # draw legend
    for i in range(proposals.shape[0]):
        legend_elements.append(Patch(facecolor=colors[i], label='Observation {}'.format(i+1)))
    if marginal_samples is not None:
        legend_elements.append(Patch(facecolor=colors[-1], label='Marg. Samples'))
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
