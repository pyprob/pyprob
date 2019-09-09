import torch
import time
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import Normal
from trandn import trandn
from truncnorm_moments import moments

n_samples = int(5)
mu = -200
std = 2.0
mus = torch.zeros(n_samples) + mu
stds = torch.zeros(n_samples) + std
lower_bound = torch.zeros(n_samples)
upper_bound = torch.zeros(n_samples) + torch.Tensor([np.infty])
samples = trandn((lower_bound-mus)/stds, (upper_bound-mus)/stds)
samples = samples*stds + mus
mean = samples.mean()

norm = Normal(0,1)

alpha = -mus/stds
t = time.time()
for _ in range(300):
    alpha_log_pdf = norm.log_prob(alpha)
print("log_prob time:", time.time()-t)
alpha_pdf = torch.exp(alpha_log_pdf)
Z = 1 - norm.cdf(alpha)
theoretical_mean = mu + std*(alpha_pdf/Z)



t = time.time()
for _ in range(1):
    logZhat, Zhat, muHat, sigmaHat, entropy = moments(lower_bound,
                                                      upper_bound,
                                                      torch.Tensor([mu]).expand_as(lower_bound),
                                                      torch.Tensor([std**2]).expand_as(lower_bound))
print("Robust time:", time.time()-t)
print("=============================\n\n")



########################################################

# PDFs

########################################################

x = 1
if x < lower_bound:
    x = lower_bound
if x > upper_bound:
    x = upper_bound

clip_a = -mu/std
clip_b = (np.infty-mu)/std
theoretical_pdf = truncnorm.pdf((x-mu)/std, clip_a, clip_b)/std
norm = Normal(0,1)
print(torch.exp(norm.log_prob((x-mu)/std)), logZhat, torch.exp(logZhat), Zhat)
robust_pdf = torch.exp(norm.log_prob((x-mu)/std)-(torch.log(torch.Tensor([std]))+logZhat))

print(f"Estimated mean: {mean}\nTheoretical mean: {theoretical_mean[0]}\nRobust evaluation of mean: {muHat.tolist()[0]}")
print("=============================\n\n")
print(f"Robust pdf: {robust_pdf[0]}\nTheoretical pdf: {theoretical_pdf}")
print("=============================")

#fig = plt.figure(figsize=(20,20))
#sns.distplot(samples, kde=False)
#plt.show()
