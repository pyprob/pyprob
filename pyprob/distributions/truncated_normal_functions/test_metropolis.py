import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from trandn_metropolis_sampler import truncnorm_sampler

n_samples = int(1e3)
mu = -3
std = 0.5
samples, mean, var, entropy = truncnorm_sampler(mu=mu, std=std, n_samples=n_samples,
                                                need_moments=True,
                                                burnin=200)

alpha = -mu/std
alpha_pdf = norm.pdf(alpha)
Z = 1 - norm.cdf(alpha)
theoretical_mean = mu + std*(alpha_pdf/Z)

print(f"Estimated mean: {mean}\nTheoretical mean: {theoretical_mean}")

fig = plt.figure(figsize=(20,20))
sns.distplot(samples, kde=False, bins=int(50))
plt.show()
