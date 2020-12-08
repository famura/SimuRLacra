import torch as to
from matplotlib import pyplot as plt

# Returns mean, upper and lower boundary of chosen confidence
# ----------------------------------------------------------------------------
def get_confidence(data, z=1.96, get_mu_sigma=False):
    # 95% confidence is calculated as: +-z*(sigma/(n)^(1/2)) where sigma
    # is the standard deviation and z=1.96 the corresponding confidence
    # interval for 95% confidence and n the number of samples
    n = data.shape[0]  # number of draws
    mu = to.mean(data, axis=0)  # calculated mean
    sigma = to.std(data, axis=0)  # calculated standard deviation
    bound = z * (sigma / n ** 0.5)
    upper = mu + bound  # upper confidence boundary
    lower = mu - bound  # lower confidence boundary
    if get_mu_sigma:
        return mu, sigma
    return mu, upper, lower


def plot_trajectories(trajectories: to.Tensor, n_parameter: int, n_observations=1, observation_data=None):
    if not trajectories.shape[1] % n_parameter:
        n_time_steps = int(trajectories.shape[1] / n_parameter)
    else:
        raise ArithmeticError("The number of trajectories has to match number of observations times number of samples")


    if not trajectories.shape[0] % n_observations:
        n_samples = int(trajectories.shape[0] / n_observations)
    else:
        raise ArithmeticError("The number of trajectories has to match number of observations times number of samples")


    mean = to.empty((n_observations, trajectories.shape[1]))
    up_bound = to.empty((n_observations, trajectories.shape[1]))
    low_bound = to.empty((n_observations, trajectories.shape[1]))

    for i in range(n_observations):
        start, end = i * n_samples, (i + 1) * n_samples
        mean[i, :], up_bound[i, :], low_bound[i, :] = get_confidence(trajectories[start:end, :])

    # plot for each observation independently
    color_list = ["lightskyblue", "bisque", "lightgreen", "salmon", "lavender"]
    mean_labels, conf_labels, observed_labels = list(), list(), list()
    for i in range(n_parameter):
        mean_labels.append("mean_" + str(i+1))
        conf_labels.append("95%-conf_" + str(i+1))
        observed_labels.append("observed_" + str(i+1))
    legend_list = mean_labels
    if observation_data is not None and observation_data.shape[0] >= n_observations:
        legend_list += observed_labels
    legend_list += conf_labels


    fig, axs = plt.subplots(n_observations, 1)
    fig.set_size_inches(9, 6)
    for cnt in range(n_observations):
        if n_observations == 1:
            ax = axs
        else:
            ax = axs[cnt]

        if cnt < n_observations - 1:
            ax.get_xaxis().set_ticklabels([])  # hide x-axis labels

        x_data = to.arange(n_time_steps)
        y_data = to.reshape(mean[cnt, :], (n_time_steps, n_parameter))
        ax.plot(x_data, y_data, '--', linewidth=2)

        upper_data = to.reshape(up_bound[cnt, :], (n_time_steps, n_parameter))
        lower_data = to.reshape(low_bound[cnt, :], (n_time_steps, n_parameter))
        for i in range(n_parameter):
            ax.fill_between(x_data, upper_data[:, i], lower_data[:, i],
                            color=color_list[i])

        if observation_data is not None and observation_data.shape[0] >= n_observations:
            y_data = to.reshape(observation_data[cnt, :], (n_time_steps, n_parameter))
            ax.plot(x_data, y_data, ':', linewidth=2, color='red')

        ax.legend(legend_list)

        cnt += 1
    plt.show()

