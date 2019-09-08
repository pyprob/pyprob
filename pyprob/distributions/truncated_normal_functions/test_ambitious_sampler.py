import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
from trandn import trandn
from truncnorm_moments import moments

n_samples = int(3)
mu = -2000.
std = 2.0
mus = torch.zeros(n_samples) + mu
stds = torch.zeros(n_samples) + std
lower_bound = torch.zeros(n_samples)
upper_bound = torch.zeros(n_samples) + torch.Tensor([np.infty])
samples = trandn((lower_bound-mus)/stds, (upper_bound-mus)/stds)
samples = samples*stds + mus
mean = samples.mean()

norm = Normal(0,1)

alpha = -mu/std
alpha_log_pdf = norm.log_prob(alpha)
alpha_pdf = torch.exp(alpha_log_pdf)
Z = 1 - norm.cdf(alpha)
theoretical_mean = mu + std*(alpha_pdf/Z)


logZhat, Zhat, muHat, sigmaHat, entropy = moments(lower_bound,
                                                  upper_bound,
                                                  torch.Tensor([mu]).expand_as(lower_bound),
                                                  torch.Tensor([std**2]).expand_as(lower_bound))


print(f"Estimated mean: {mean}\nTheoretical mean: {theoretical_mean}\nRobust evaluation of mean: {muHat.item()}")

fig = plt.figure(figsize=(20,20))
sns.distplot(samples, kde=False)
plt.show()
