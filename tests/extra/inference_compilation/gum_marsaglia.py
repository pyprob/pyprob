import os
import shutil
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

import pyprob
from pyprob import Model, InferenceEngine
from pyprob.distributions import Uniform, Normal


class GaussianWithUnknownMeanMarsaglia(Model):
    def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2), replace=True, *args, **kwargs):
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev
        self.likelihood_stddev = likelihood_stddev
        self.replace = replace
        super().__init__('Gaussian with unknown mean (Marsaglia)', *args, **kwargs)

    def marsaglia(self, mean, stddev):
        uniform = Uniform(-1, 1)
        s = 1
        i = 0
        while True:
            x = pyprob.sample(uniform, replace=self.replace)
            y = pyprob.sample(uniform, replace=self.replace)
            s = x*x + y*y
            i += 1
            if float(s) < 1:
                pyprob.tag(x, name='x_accepted')
                pyprob.tag(y, name='y_accepted')
                pyprob.tag(s, name='s_accepted')
                break
            else:
                pyprob.tag(x, name='x_rejected')
                pyprob.tag(y, name='y_rejected')
                pyprob.tag(s, name='s_rejected')
        pyprob.tag(i, name='iterations')
        return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

    def forward(self):
        mu = self.marsaglia(self.prior_mean, self.prior_stddev)
        likelihood = Normal(mu, self.likelihood_stddev)
        pyprob.tag(mu, name='mu')
        pyprob.observe(likelihood, name='obs0')
        pyprob.observe(likelihood, name='obs1')
        return mu


def produce_results(results_dir):
    train_traces_max = 500000
    train_traces_resolution = 10
    infer_traces = 10000
    train_traces_step = int(train_traces_max/train_traces_resolution)
    observe = {'obs0': 8, 'obs1': 9}

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    pyprob.util.create_path(results_dir, directory=True)

    model_replace_true = GaussianWithUnknownMeanMarsaglia(replace=True)
    model_replace_false = GaussianWithUnknownMeanMarsaglia(replace=False)

    model_replace_true_is = model_replace_true.posterior_distribution(infer_traces, observe=observe)
    model_replace_false_is = model_replace_false.posterior_distribution(infer_traces, observe=observe)

    traces = []
    model_replace_true_loss = []
    model_replace_false_loss = []
    model_replace_true_is_ess = float(model_replace_true_is.effective_sample_size) / infer_traces
    model_replace_false_is_ess = float(model_replace_false_is.effective_sample_size) / infer_traces
    model_replace_true_ic_ess = []
    model_replace_false_ic_ess = []
    model_replace_true.learn_inference_network(0, batch_size=1, observe_embeddings={'obs0': {}, 'obs1': {}})
    model_replace_false.learn_inference_network(0, batch_size=1, observe_embeddings={'obs0': {}, 'obs1': {}})

    for train_traces in range(0, train_traces_max+1, train_traces_step):
        print('\ntrain_traces: {}/{}'.format(train_traces, train_traces_max))
        model_replace_true.learn_inference_network(train_traces_step)
        model_replace_false.learn_inference_network(train_traces_step)
        model_replace_true_loss.append(float(model_replace_true._inference_network._history_train_loss[-1]))
        model_replace_false_loss.append(float(model_replace_false._inference_network._history_train_loss[-1]))
        model_replace_true_ic = model_replace_true.posterior_distribution(infer_traces, observe=observe, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK)
        model_replace_false_ic = model_replace_false.posterior_distribution(infer_traces, observe=observe, inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK)
        model_replace_true_ic_ess.append(float(model_replace_true_ic.effective_sample_size))
        model_replace_false_ic_ess.append(float(model_replace_false_ic.effective_sample_size))
        traces.append(train_traces)

    model_replace_true_ic_ess = np.array(model_replace_true_ic_ess) / infer_traces
    model_replace_false_ic_ess = np.array(model_replace_false_ic_ess) / infer_traces

    plot_file_name = os.path.join(results_dir, 'result.pdf')
    print('Saving result plot to: {}'.format(plot_file_name))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    ax1.hlines(model_replace_true_is_ess, xmin=0, xmax=train_traces_max, label='IS, Replace=True')
    ax1.hlines(model_replace_false_is_ess, xmin=0, xmax=train_traces_max, label='IS, Replace=False')
    ax1.plot(traces, model_replace_true_ic_ess, label='IC, Replace=True')
    ax1.plot(traces, model_replace_false_ic_ess, label='IC, Replace=False')
    ax1.legend(loc='best')
    ax1.set_ylabel('Normalized ESS')
    ax2.plot(traces, model_replace_true_loss, label='Replace=True')
    ax2.plot(traces, model_replace_false_loss, label='Replace=False')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Training traces')
    ax2.legend(loc='best')
    plt.tight_layout()
    fig.savefig(plot_file_name)


if __name__ == '__main__':
    pyprob.set_random_seed(1)
    pyprob.set_cuda(False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print('Current dir: {}'.format(current_dir))

    results_dir = os.path.join(current_dir, 'gum_marsaglia')
    produce_results(results_dir=results_dir)

    print('Done')
