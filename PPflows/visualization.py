import os

import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt


def scatter_points(points, directory, iteration, flow_length):

    X_LIMS = (-7, 7)
    Y_LIMS = (-7, 7)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25)
    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_title(
        "Flow length: {}\n Samples on iteration #{}"
        .format(flow_length, iteration)
    )

    fig.savefig(os.path.join(directory, "flow_result_{}.png".format(iteration)))
    plt.close()


def plot_density(density, directory):

    X_LIMS = (-7, 7)
    Y_LIMS = (-7, 7)

    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    x1, x2 = np.meshgrid(x1, x2)
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()

    z = np.c_[x1, x2]
    z = torch.FloatTensor(z)
    z = Variable(z)

    density_values = density(z).data.numpy().reshape(shape)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(density_values, extent=(*X_LIMS, *Y_LIMS), cmap="summer")
    ax.set_title("True density")

    fig.savefig(os.path.join(directory, "density.png"))
    plt.close()
