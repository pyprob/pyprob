import unittest
import time
import math
import sys
from termcolor import colored

import pyprob
from pyprob import util, Model
from pyprob.distributions import Normal


importance_sampling_samples = 5000
importance_sampling_kl_divergence = 0
importance_sampling_duration = 0

inference_compilation_samples = 5000
inference_compilation_kl_divergence = 0
inference_compilation_duration = 0
inference_compilation_training_traces = 20000

lightweight_metropolis_hastings_samples = 5000
lightweight_metropolis_hastings_kl_divergence = 0
lightweight_metropolis_hastings_duration = 0

random_walk_metropolis_hastings_samples = 5000
random_walk_metropolis_hastings_kl_divergence = 0
random_walk_metropolis_hastings_duration = 0

def add_importance_sampling_kl_divergence(val):
    global importance_sampling_kl_divergence
    importance_sampling_kl_divergence += val


def add_inference_compilation_kl_divergence(val):
    global inference_compilation_kl_divergence
    inference_compilation_kl_divergence += val


def add_lightweight_metropolis_hastings_kl_divergence(val):
    global lightweight_metropolis_hastings_kl_divergence
    lightweight_metropolis_hastings_kl_divergence += val


def add_random_walk_metropolis_hastings_kl_divergence(val):
    global random_walk_metropolis_hastings_kl_divergence
    random_walk_metropolis_hastings_kl_divergence += val


def add_importance_sampling_duration(val):
    global importance_sampling_duration
    importance_sampling_duration += val


def add_inference_compilation_duration(val):
    global inference_compilation_duration
    inference_compilation_duration += val


def add_lightweight_metropolis_hastings_duration(val):
    global lightweight_metropolis_hastings_duration
    lightweight_metropolis_hastings_duration += val


def add_random_walk_metropolis_hastings_duration(val):
    global random_walk_metropolis_hastings_duration
    random_walk_metropolis_hastings_duration += val


class GaussianWithUnknownMeanTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        class GaussianWithUnknownMean(Model):
            def __init__(self, prior_mean=1, prior_stddev=math.sqrt(5), likelihood_stddev=math.sqrt(2)):
                self.prior_mean = prior_mean
                self.prior_stddev = prior_stddev
                self.likelihood_stddev = likelihood_stddev
                super().__init__('Gaussian with unknown mean')

            def forward(self):
                mu = pyprob.sample(Normal(self.prior_mean, self.prior_stddev))
                likelihood = Normal(mu, self.likelihood_stddev)
                pyprob.observe(likelihood, name='obs1')
                pyprob.observe(likelihood, name='obs2')
                return mu

        self._model = GaussianWithUnknownMean()
        super().__init__(*args, **kwargs)

    def test_inference_gum_posterior_importance_sampling(self):
        samples = importance_sampling_samples
        true_posterior = Normal(7.25, math.sqrt(1/1.2))
        posterior_mean_correct = float(true_posterior.mean)
        posterior_stddev_correct = float(true_posterior.stddev)
        prior_mean_correct = 1.
        prior_stddev_correct = math.sqrt(5)

        start = time.time()
        posterior = self._model.posterior_distribution(samples, observe={'obs1': 8, 'obs2': 9})
        add_importance_sampling_duration(time.time() - start)

        posterior_mean = float(posterior.mean)
        posterior_mean_unweighted = float(posterior.unweighted().mean)
        posterior_stddev = float(posterior.stddev)
        posterior_stddev_unweighted = float(posterior.unweighted().stddev)
        kl_divergence = float(pyprob.distributions.Distribution.kl_divergence(true_posterior, Normal(posterior.mean, posterior.stddev)))

        util.debug('samples', 'prior_mean_correct', 'posterior_mean_unweighted', 'posterior_mean', 'posterior_mean_correct', 'prior_stddev_correct', 'posterior_stddev_unweighted', 'posterior_stddev', 'posterior_stddev_correct', 'kl_divergence')
        add_importance_sampling_kl_divergence(kl_divergence)

        self.assertAlmostEqual(posterior_mean_unweighted, prior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev_unweighted, prior_stddev_correct, places=0)
        self.assertAlmostEqual(posterior_mean, posterior_mean_correct, places=0)
        self.assertAlmostEqual(posterior_stddev, posterior_stddev_correct, places=0)
        self.assertLess(kl_divergence, 0.25)


if __name__ == '__main__':
    # if torch.cuda.is_available():
        # pyprob.set_cuda(True)
    pyprob.set_verbosity(2)
    tests = []
    tests.append('GaussianWithUnknownMeanTestCase')
    # tests.append('GaussianWithUnknownMeanMarsagliaTestCase')
    # tests.append('HiddenMarkovModelTestCase')
    # tests.append('BranchingTestCase')

    time_start = time.time()
    success = unittest.main(defaultTest=tests, verbosity=2, exit=False).result.wasSuccessful()
    print('\nDuration                   : {}'.format(util.days_hours_mins_secs_str(time.time() - time_start)))
    print('Models run                 : {}'.format(' '.join(tests)))
    print('\nTotal inference performance:\n')
    print(colored('                                 Samples        KL divergence  Duration (s) ', 'yellow', attrs=['bold']))
    print(colored('Importance sampling            : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(importance_sampling_samples, importance_sampling_kl_divergence, importance_sampling_duration), 'white', attrs=['bold']))
    print(colored('Inference compilation          : ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(inference_compilation_samples, inference_compilation_kl_divergence, inference_compilation_duration), 'white', attrs=['bold']))
    print(colored('Lightweight Metropolis Hastings: ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}'.format(lightweight_metropolis_hastings_samples, lightweight_metropolis_hastings_kl_divergence, lightweight_metropolis_hastings_duration), 'white', attrs=['bold']))
    print(colored('Random-walk Metropolis Hastings: ', 'yellow', attrs=['bold']), end='')
    print(colored('{:+.6e}  {:+.6e}  {:+.6e}\n'.format(random_walk_metropolis_hastings_samples, random_walk_metropolis_hastings_kl_divergence, random_walk_metropolis_hastings_duration), 'white', attrs=['bold']))
    sys.exit(0 if success else 1)
