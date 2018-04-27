import argparse

import torch
from torch.autograd import Variable
from torch import optim
from pyprob.PPflows.experiments import experiment

from pyprob.PPflows.visualization import plot_density, scatter_points
from pyprob.PPflows.utils import random_normal_samples
from pyprob.PPflows.flow import NormalizingFlow
from pyprob.PPflows.losses import FreeEnergyBound
from pyprob.PPflows.densities import p_z


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--log_interval", type=int, default=300,
    help="How frequenlty to print the training stats."
)
parser.add_argument(
    "--plot_interval", type=int, default=300,
    help="How frequenlty to plot samples from current distribution."
)
parser.add_argument(
    "--plot_points", type=int, default=1000,
    help="How many to points to generate for one plot."
)

args = parser.parse_args()

torch.manual_seed(42)


with experiment({
    "batch_size": 40,
    "iterations": 10000,
    "initial_lr": 0.01,
    "lr_decay": 0.999,
    "flow_length": 16,
    "name": "planar"
}) as experiment:
    config = experiment
    experiment.register_directory("samples")
    experiment.register_directory("distributions")

    flow = NormalizingFlow(dim=2, flow_length=config.flow_length)
    bound = FreeEnergyBound(density=p_z)
    print('Debug flow.parameters(), will call nn.NormalizingFlow.nn.parameters() : {0}'.format(flow.parameters()))
    optimizer = optim.RMSprop(flow.parameters(), lr=config.initial_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)

    plot_density(p_z, directory=experiment.distributions)

    def should_log(iteration):
        return iteration % args.log_interval == 0

    def should_plot(iteration):
        return iteration % args.plot_interval == 0

    for iteration in range(1, config.iterations + 1):

        scheduler.step()

        samples = Variable(random_normal_samples(config.batch_size))
        zk, log_jacobians = flow(samples)

        optimizer.zero_grad()
        loss = bound(zk, log_jacobians)
        loss.backward()
        optimizer.step()

        if should_log(iteration):
            print("Loss on iteration {}: {}".format(iteration , loss.data[0]))

        if should_plot(iteration):
            samples = Variable(random_normal_samples(args.plot_points))
            zk, det_grads = flow(samples)
            scatter_points(
                zk.data.numpy(),
                directory=experiment.samples,
                iteration=iteration,
                flow_length=config.flow_length
            )
