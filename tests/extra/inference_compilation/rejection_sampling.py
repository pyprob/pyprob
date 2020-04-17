import os
import math
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt

import pyprob
from pyprob import Model
from pyprob.distributions import Uniform, Normal


class GaussianWithUnknownMean(Model):
    def __init__(self):
        super().__init__('Gaussian with unknown mean')
        self.prior_mean = 0.
        self.prior_stddev = 1.
        self.likelihood_stddev = math.sqrt(0.2)
        self.prior_true = Normal(self.prior_mean, self.prior_stddev)

    def posterior_true(self, obs):
        n = len(obs)
        posterior_var = 1/(n/self.likelihood_stddev**2 + 1/self.prior_stddev**2)
        posterior_mu = posterior_var * (self.prior_mean/self.prior_stddev**2 + n*np.mean(obs)/self.likelihood_stddev**2)
        return Normal(posterior_mu, math.sqrt(posterior_var))

    def rejection_sampling(self):
        u = pyprob.sample(Uniform(0, 1), control=False)
        if u > 0.5:
            while True:
                x = pyprob.sample(Normal(self.prior_mean, self.prior_stddev * 4), replace=True)
                u2 = pyprob.sample(Uniform(0, 1), control=False)
                if x < 0 and u2 < 0.25 * torch.exp(Normal(self.prior_mean, self.prior_stddev).log_prob(x) - Normal(self.prior_mean, self.prior_stddev*4).log_prob(x)):
                    return x
        else:
            while True:
                x = pyprob.sample(Normal(self.prior_mean, self.prior_stddev), replace=True)
                if x >= 0:
                    return x

    def forward(self):
        mu = self.rejection_sampling()
        likelihood = Normal(mu, self.likelihood_stddev)
        pyprob.observe(likelihood, name='obs0')
        return mu


def produce_results(results_dir):
    infer_traces = 2500
    train_traces = 10000
    model = GaussianWithUnknownMean()

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    pyprob.util.create_path(results_dir, directory=True)

    prior_samples = model.prior_results(num_traces=infer_traces)
    fig = plt.figure(figsize=(10, 5))
    model.prior_true.plot(label='True prior', min_val=-5, max_val=5, show=False, fig=fig)
    prior_samples.plot_histogram(label='Empirical prior', alpha=0.75, show=False, bins=50, fig=fig)
    plt.legend()
    prior_plot_file_name = os.path.join(results_dir, 'prior.pdf')
    plt.savefig(prior_plot_file_name)

    is_posterior_samples = model.posterior_results(num_traces=infer_traces, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING, observe={'obs0': 0.})
    fig = plt.figure(figsize=(10, 5))
    model.posterior_true([0.]).plot(label='True posterior', min_val=-5, max_val=5, show=False, fig=fig)
    is_posterior_samples.unweighted().plot_histogram(label='Empirical proposal', alpha=0.75, show=False, bins=50, fig=fig)
    is_posterior_samples.plot_histogram(label='Empirical posterior', alpha=0.75, show=False, bins=50, fig=fig)
    plt.legend()
    is_posterior_plot_file_name = os.path.join(results_dir, 'posterior_IS.pdf')
    plt.savefig(is_posterior_plot_file_name)

    model.learn_inference_network(num_traces=train_traces, observe_embeddings={'obs0' : {'dim' : 32}}, inference_network=pyprob.InferenceNetwork.LSTM)

    ic_iw0_posterior_samples = model.posterior_results(num_traces=infer_traces, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 0.}, importance_weighting=pyprob.ImportanceWeighting.IW0)
    fig = plt.figure(figsize=(10, 5))
    model.posterior_true([0.]).plot(label='True posterior', min_val=-5, max_val=5, show=False, fig=fig)
    ic_iw0_posterior_samples.unweighted().plot_histogram(label='Empirical proposal', alpha=0.75, show=False, bins=50, fig=fig)
    ic_iw0_posterior_samples.plot_histogram(label='Empirical posterior', alpha=0.75, show=False, bins=50, fig=fig)
    plt.legend()
    ic_iw0_posterior_plot_file_name = os.path.join(results_dir, 'posterior_IC_IW0.pdf')
    plt.savefig(ic_iw0_posterior_plot_file_name)

    ic_iw1_posterior_samples = model.posterior_results(num_traces=infer_traces, inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, observe={'obs0': 0.}, importance_weighting=pyprob.ImportanceWeighting.IW1)
    fig = plt.figure(figsize=(10, 5))
    model.posterior_true([0.]).plot(label='True posterior', min_val=-5, max_val=5, show=False, fig=fig)
    ic_iw1_posterior_samples.unweighted().plot_histogram(label='Empirical proposal', alpha=0.75, show=False, bins=50, fig=fig)
    ic_iw1_posterior_samples.plot_histogram(label='Empirical posterior', alpha=0.75, show=False, bins=50, fig=fig)
    plt.legend()
    ic_iw1_posterior_plot_file_name = os.path.join(results_dir, 'posterior_IC_IW1.pdf')
    plt.savefig(ic_iw1_posterior_plot_file_name)


if __name__ == '__main__':
    pyprob.seed(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('Current dir: {}'.format(current_dir))

    results_dir = os.path.join(current_dir, 'rejection_sampling')
    produce_results(results_dir=results_dir)

    print('Done')
