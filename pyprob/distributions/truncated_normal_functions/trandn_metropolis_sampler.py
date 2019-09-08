import numpy as np
from scipy.stats import norm

def estimate_entropy(samples, mu, std):
    raise NotImplementedError()

def gam(mu, std, x):
    precision = 1/(std**2)
    return 0.5*precision*(np.exp(2*x)-2*mu*np.exp(x)) - x

def log_acceptance_rate(xold, xstar, mu, std):
    gamstar = gam(mu, std, xstar)
    gamold = gam(mu, std, xold)
    logr = gamold - gamstar
    return logr

def sample_w_moments(mu, std, xold, n_samples, burnin):

    xold = xold
    samples = np.empty(n_samples-burnin)
    sigma = 1
    entry = 0
    n_accepted = 0.0
    for i in range(n_samples):
        #xstar = norm.rvs(loc=xold, scale=sigma)
        xstar = np.random.randn()
        xstar = xstar*sigma + xold
        logr = log_acceptance_rate(xold, xstar, mu, std)
        if logr >= 0:
            n_accepted += 1
            xold = xstar
        elif np.random.uniform() < np.exp(logr):
            n_accepted += 1
            xold = xstar
        else:
            pass

        if i < burnin and i%20==0: # adjust sigma during burin
            rate = n_accepted/20
            if rate < 0.6:
                # if acceptance rate is low decrease sigma for smaller steps
                sigma = sigma - 0.1
            elif rate > 0.8: #
                # if acceptance rate is high, increase sigma for larger steps
                sigma = sigma + 0.1
            else:
                pass
            n_accepted = 0

        if i >= burnin:
            samples[entry] = np.exp(xold)
            entry += 1

    entropy = estimate_entropy(samples, mu, std)

    return samples, samples.mean(), samples.var(), entropy

def sample_without_moments(mu, std, xold, n_samples, burnin):

    xold = xold
    sigma = 1
    n_accepted = 0.0
    for i in range(n_samples):
        xstar = norm.rvs(mu=xold, scale=sigma)
        if xstar >= 0:
            logr = log_acceptance_rate(xold, xstar, mu, std)
            if logr >= 0:
                n_accepted += 1
                xold = xstar
            elif np.random.uniform() < np.exp(logr):
                n_accepted += 1
                xold = xstar
            else:
                pass

        if i < burnin and i%20==0: # adjust sigma during burin
            rate = n_accepted/20
            if rate < 0.6:
                # if acceptance rate is low decrease sigma for smaller steps
                sigma = sigma - 0.1
            elif rate > 0.8: #
                # if acceptance rate is high, increase sigma for larger steps
                sigma = sigma + 0.1
            else:
                pass
            n_accepted = 0


    return np.exp(xold)

def truncnorm_sampler(mu=0, std=1, init=1, n_samples=2000, burnin=200,
                      need_moments=False):
    """
        Metropolis sampler, for the truncated normal bounded below by a=0
        and unbounded above

        - mu is the mean of the parent normal
        - std is the standard deviation of the parent normal
    """

    if not need_moments:
        return sample_without_moments(mu, std, init, n_samples, burnin)
    else:
        samples, mean, var, etrpy = sample_w_moments(mu, std, init, n_samples,
                                                     burnin)
        return samples, mean, var, etrpy
