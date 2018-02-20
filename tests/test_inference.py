import unittest
import sys
import math
import torch
import torch.nn.functional as F
from termcolor import colored
import time

import pyprob
from pyprob import util
from pyprob import Model
from pyprob.distributions import Categorical, Empirical, Normal, Uniform


samples = 1000
training_traces = 5000
perf_score_importance_sampling = 0
perf_score_inference_compilation = 0

def add_perf_score_importance_sampling(score):
    global perf_score_importance_sampling
    perf_score_importance_sampling += score

def add_perf_score_inference_compilation(score):
    global perf_score_inference_compilation
    perf_score_inference_compilation += score


class GaussianWithUnknownMeanTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean')

            def forward(self, observation=[]):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_inference_gum_posterior_importance_sampling(self):
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.15)
        add_perf_score_importance_sampling(kl_divergence)

    def test_inference_gum_posterior_inference_compilation(self):
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        self._model.learn_inference_network(observation=[1,1], early_stop_traces=training_traces)
        posterior = self._model.posterior_distribution(samples, use_inference_network=True, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.15)
        add_perf_score_inference_compilation(kl_divergence)


class GaussianWithUnknownMeanMarsagliaTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class GaussianWithUnknownMeanMarsaglia(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean (Marsaglia)')

            def marsaglia(self, mean, stddev):
                uniform = Uniform(-1, 1)
                s = 1
                while float(s) >= 1:
                    x = pyprob.sample(uniform)
                    y = pyprob.sample(uniform)
                    s = x*x + y*y
                return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

            def forward(self, observation=[]):
                mu = self.marsaglia(self.prior_mean, self.prior_stddev)
                likelihood = Normal(mu, self.likelihood_stddev)
                for o in observation:
                    pyprob.observe(likelihood, o)
                return mu

        self._model = GaussianWithUnknownMeanMarsaglia()
        super().__init__(*args, **kwargs)

    def test_inference_gum_marsaglia_posterior_importance_sampling(self):
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)
        add_perf_score_importance_sampling(kl_divergence)

    def test_inference_gum_marsaglia_posterior_inference_compilation(self):
        observation = [8,9]
        posterior_mean_correct = 7.25
        posterior_stddev_correct = math.sqrt(1/1.2)

        self._model.learn_inference_network(observation=[1,1], early_stop_traces=training_traces, learning_rate=0.0001)
        posterior = self._model.posterior_distribution(samples, use_inference_network=True, observation=observation)
        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.mean_unweighted)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.stddev_unweighted)
        kl_divergence = float(util.kl_divergence_normal(posterior_mean_correct, posterior_stddev_correct, posterior.mean, posterior_stddev))

        util.debug('training_traces', 'samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')

        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)
        add_perf_score_inference_compilation(kl_divergence)


class HiddenMarkovModelTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf
        class HiddenMarkovModel(Model):
            def __init__(self, init_dist, trans_dists, obs_dists):
                self.init_dist = init_dist
                self.trans_dists = trans_dists
                self.obs_dists = obs_dists
                super().__init__('Hidden Markov model')

            def forward(self, observation=[]):
                states = [pyprob.sample(init_dist)]
                for o in observation:
                    state = pyprob.sample(self.trans_dists[int(states[-1])])
                    pyprob.observe(self.obs_dists[int(state)], o)
                    states.append(state)
                return torch.stack([util.one_hot(3, int(s)) for s in states])

        init_dist = Categorical([1, 1, 1])
        trans_dists = [Categorical([0.1, 0.5, 0.4]),
                       Categorical([0.2, 0.2, 0.6]),
                       Categorical([0.15, 0.15, 0.7])]
        obs_dists = [Normal(-1, 1),
                     Normal(1, 1),
                     Normal(0, 1)]
        self._model = HiddenMarkovModel(init_dist, trans_dists, obs_dists)

        self._observation = [0.9, 0.8, 0.7, 0.0, -0.025, -5.0, -2.0, -0.1, 0.0, 0.13, 0.45, 6, 0.2, 0.3, -1, -1]
        self._posterior_mean_correct = util.to_variable([[0.3775, 0.3092, 0.3133],
                                                         [0.0416, 0.4045, 0.5539],
                                                         [0.0541, 0.2552, 0.6907],
                                                         [0.0455, 0.2301, 0.7244],
                                                         [0.1062, 0.1217, 0.7721],
                                                         [0.0714, 0.1732, 0.7554],
                                                         [0.9300, 0.0001, 0.0699],
                                                         [0.4577, 0.0452, 0.4971],
                                                         [0.0926, 0.2169, 0.6905],
                                                         [0.1014, 0.1359, 0.7626],
                                                         [0.0985, 0.1575, 0.7440],
                                                         [0.1781, 0.2198, 0.6022],
                                                         [0.0000, 0.9848, 0.0152],
                                                         [0.1130, 0.1674, 0.7195],
                                                         [0.0557, 0.1848, 0.7595],
                                                         [0.2017, 0.0472, 0.7511],
                                                         [0.2545, 0.0611, 0.6844]])
        super().__init__(*args, **kwargs)

    def test_inference_hmm_posterior_importance_sampling(self):
        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        posterior = self._model.posterior_distribution(samples, observation=observation)
        posterior_mean_unweighted = posterior.mean_unweighted
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance')

        self.assertLess(l2_distance, 4)
        add_perf_score_importance_sampling(l2_distance)

    def test_inference_hmm_posterior_inference_compilation(self):
        observation = self._observation
        posterior_mean_correct = self._posterior_mean_correct

        self._model.learn_inference_network(observation=torch.zeros(16,3), early_stop_traces=training_traces, learning_rate=0.0001)
        posterior = self._model.posterior_distribution(samples, use_inference_network=True, observation=observation)
        posterior_mean_unweighted = posterior.mean_unweighted
        posterior_mean = posterior.mean

        l2_distance = float(F.pairwise_distance(posterior_mean, posterior_mean_correct).sum())

        util.debug('samples', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'l2_distance')

        self.assertLess(l2_distance, 4)
        add_perf_score_inference_compilation(l2_distance)


if __name__ == '__main__':
    # if torch.cuda.is_available():
        # pyprob.set_cuda(True)
    tests = []
    # tests.append('GaussianWithUnknownMeanTestCase')
    tests.append('GaussianWithUnknownMeanMarsagliaTestCase')
    # tests.append('HiddenMarkovModelTestCase')

    time_start = time.time()
    success = unittest.main(defaultTest=tests, verbosity=2, exit=False).result.wasSuccessful()
    print('\nDuration             : {}'.format(util.days_hours_mins_secs_str(time.time() - time_start)))
    print('Models run           : {}'.format(' '.join(tests)))
    print('Samples              : {}'.format(samples))
    print('Training traces      : {}\n'.format(training_traces))
    print('\nTotal inference performance scores\n')
    print(colored('Importance sampling  : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}'.format(perf_score_importance_sampling), 'white', attrs=['bold']))
    print(colored('Inference compilation: ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}\n'.format(perf_score_inference_compilation), 'white', attrs=['bold']))
    sys.exit(0 if success else 1)
