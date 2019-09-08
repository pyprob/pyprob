import sys
import numpy as np
from scipy.special import erfcinv
from torch.distributions import Uniform, Normal
import torch

expm1 = torch.expm1
erfc = torch.erfc
uniform = Uniform(0,1)
normal = Normal(0,1)

def trandn(l,u):
    ## truncated normal generator
    # * efficient generator of a vector of length(l)=length(u)
    # from the standard multivariate normal distribution,
    # truncated over the region [l,u];
    # infinite values for 'u' and 'l' are accepted;
    # * Remark:
    # If you wish to simulate a random variable
    # 'Z' from the non-standard Gaussian N(m,s^2)
    # conditional on l<Z<u, then first simulate
    # X=trandn((l-m)/s,(u-m)/s) and set Z=m+s*X;

    # Reference:
    # Botev, Z. I. (2016). "The normal law under linear restrictions:
    # simulation and estimation via minimax tilting". Journal of the
    # Royal Statistical Society: Series B (Statistical Methodology).
    # doi:10.1111/rssb.12162
    l = l.flatten()
    u = u.flatten() # make 'l' and 'u' column vectors

    if len(l) != len(u):
        print('Truncation limits have to be vectors of the same length')
        sys.exit()

    x = torch.empty(len(l))
    a = .66 # treshold for switching between methods

    # three cases to consider:
    # case 1: a<l<u

    I = l>a
    if I.any():
        tl=l[I]
        tu=u[I]
        x[I]=ntail(tl,tu)

    # case 2: l<u<-a

    J = u<-a
    if J.any():
        tl=-u[J]
        tu=-l[J]
        x[J] = -ntail(tl,tu)

    # case 3: otherwise use inverse transform or accept-reject

    I=~(I|J);
    if I.any():
        tl=l.masked_select(I)
        tu=u.masked_select(I)
        x[I]=tn(tl,tu)

    return x

#################################################################

def ntail(l,u):

    # samples a column vector of length=length(l)=length(u)
    # from the standard multivariate normal distribution,
    # truncated over the region [l,u], where l>0 and
    # l and u are column vectors;
    # uses acceptance-rejection from Rayleigh distr.
    # similar to Marsaglia (1964);

    c = l**2/2
    n = torch.Size([len(l)])
    f = expm1(c-u**2/2)

    x = c - np.log(1+uniform.sample(n)*f); # sample using Rayleigh

    # keep list of rejected

    I = uniform.sample(n)**2*x > c

    while I.any(): # while there are rejections
        cy = c[I] # find the thresholds of rejected
        tmp_shape = torch.Size([len(cy)])
        y = cy - torch.log(1+uniform.sample(tmp_shape)*f[I])
        idx = (uniform.sample(tmp_shape)**2)*y < cy # accepted
        tmp = I.clone()
        I[tmp] = idx # make the list of elements in x to update
        x[I] = y[idx] # store the accepted
        I[tmp] = ~idx # remove accepted from list

#    while d>0: # while there are rejections
#        cy = c[I] # find the thresholds of rejected
#        y = cy - np.log(1+np.random.uniform(size=d)*f[I])
#        idx = (np.random.uniform(size=d)**2)*y < cy # accepted
#        x[I[idx]] = y[idx] # store the accepted
#        I = I[~idx] # remove accepted from list
#        d = len(I) # number of rejected

    x = torch.sqrt(2*x); # this Rayleigh transform can be delayed till the end

    return x

##################################################################

def tn(l,u):

    # samples a column vector of length=length(l)=length(u)
    # from the standard multivariate normal distribution,
    # truncated over the region [l,u], where -a<l<u<a for some
    # 'a' and l and u are column vectors;
    # uses acceptance rejection and inverse-transform method;
    tol = 2 # controls switch between methods

    # threshold can be tuned for maximum speed for each platform
    # case: abs(u-l)>tol, uses accept-reject from randn

    I = torch.abs(u-l)>tol
    x = l

    if I.any():
        tl=l[I]
        tu=u[I]
        x[I]=trnd(tl,tu)

    # case: abs(u-l)<tol, uses inverse-transform

    I=~I
    if I.any():

        tl=l[I]
        tu=u[I]
        pl = erfc(tl/torch.sqrt(2))/2
        pu = erfc(tu/torch.sqrt(2))/2

        x[I] = torch.sqrt(2)*erfcinv(2*(pl-(pl-pu)*uniform.sample(len(tl))))

    return x

#############################################################

def trnd(l,u):

    # uses acceptance rejection to simulate from truncated normal
    n = torch.Size([len(l)])
    x=normal.sample(n) # sample normal

    # keep list of rejected
    I = (x<l)|(x>u)
    while I.any(): # while there are rejections
        ly = l[I] # find the thresholds of rejected
        uy = u[I]
        tmp_shape = torch.Size([len(ly)])
        y = normal.sample(tmp_shape)
        idx = (y>ly)&(y<uy) # accepted
        tmp = I.clone()
        I[tmp] = idx # make the list of elements in x to update
        x[I] = y[idx] # store the accepted
        I[tmp] = ~idx # remove accepted from list

#    d = len(I)
#    while d>0: # while there are rejections
#        ly = l[I] # find the thresholds of rejected
#        uy = u[I]
#        y = np.random.randn(len(ly))
#        idx = np.logical_and(y>ly,y<uy) # accepted
#        x[I[idx]] = y[idx] # store the accepted
#        I = I[~idx] # remove accepted from list
#        d = len(I) # number of rejected

    return x
