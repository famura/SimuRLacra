"""
...plotting log probability for comparison

"""
from matplotlib import pyplot as plt
import numpy as np


def add_plot(posterior, samples, x_o, i):
    log_probability = posterior.log_prob(samples, x=x_o)  # , norm_posterior = False)
    print(log_probability)
    print(log_probability.shape)

    plt.plot(i, log_probability, "or")
    plt.ylabel("log_prob")
    plt.xlabel("num_sim")
    plt.legend()


def plot_log():
    plt.show()
