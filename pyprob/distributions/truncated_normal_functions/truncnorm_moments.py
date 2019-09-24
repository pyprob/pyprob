import torch
import numpy as np
from .utils import erfcx

#############################################

## used for running the test_ambitious_sampler.py script

# from utils.utils import erfcx
# class util:
#     _device = torch.device('cpu')

#############################################

"""
# John P Cunningham
# 2010
#
# Credit given to https://github.com/cunni/epmgp
#
# Calculates the truncated zeroth, first, and second moments of a
# univariate normal distribution.
#
# Much special care is taken to ensure numerical stability over a wide
# range of values, which is particularly important when calculating tail
# probabilities.
#
# there is use of the scaled complementary error function erfcx, etc.
# The actual equations can be found in Jawitz 2004 or Cunningham PhD thesis
# 2009 (Chap 4, Eq 4.6 (note typo in first... alpha and beta are swapped)).
# -This is to be used by Cunningham and Hennig, EP MGP paper.
#
# KEY: normcdf and erf and erfc are unstable when their arguments get big.
# If we are interested in tail probabilities (we are), and if we care about
# logZ more than Z (we do), then normcf/erf/erfc are limited in their
# usefulness.  Instead we can use erfcx(z) = exp(z^2)erfc(z), which has
# some foibles of its own.  The key is to consider that erfcx is stable close to erfc==0, but
# less stable close to erfc==1.  Since the Gaussian is symmetric
# about 0, we can choose to flip the calculation around the origin
# to improve our stability.  For example, if a is -inf, and b is
# -100, then a naive application will return 0.  However, we can
# just flip this around to be erfc(b)= erfc(100), which we can then
# STABLY calculate with erfcx(100)...neat.  This leads to many
# different cases, when either argument is inf or not.
# Also there may appear to be some redundancy here, but it is also
# worth noting that a less detailed application of erfcx can be
# troublesome, as erfcx(-big numbers) = Inf, which wreaks havoc on
# a lot of the calculations.  The cases below treat this when
# necessary.
#
# NOTE: works with singletons or vectors only!
#
# The code is very fast as stands, so loops and other cases are used
# (instead of vectorizing) so that things are most readable.
"""

def moments(lowerB, upperB, mu, sigma):
    device = mu.device

    lowerB = lowerB.double()
    upperB = upperB.double()
    mu = mu.double()
    sigma = sigma.double()

    pi = torch.Tensor([np.pi]).double().expand_as(mu).to(device=device)
    logZhat = torch.empty_like(mu).double().to(device=device)
    Zhat = torch.empty_like(mu).double().to(device=device)
    muHat = torch.empty_like(mu).double().to(device=device)
    sigmaHat = torch.empty_like(mu).double().to(device=device)
    entropy = torch.empty_like(mu).double().to(device=device)
    meanConst = torch.empty_like(mu).double().to(device=device)
    varConst = torch.empty_like(mu).double().to(device=device)

    """
    lowerB is the lower bound
    upperB is the upper bound
    mu is the mean of the normal distribution (which is truncated)
    sigma is the VARIANCE of the normal distribution (which is truncated)

    """
    """
    # establish bounds
    """
    a = (lowerB - mu)/(torch.sqrt(2*sigma))
    b = (upperB - mu)/(torch.sqrt(2*sigma))


    """
    # do the stable calculation
    """
    # written out in long format to make clear the steps.  There are steps to
    # take to make this code shorter, but I think it is most readable this way.
    # KEY: The key is to consider that erfcx is stable close to erfc==0, but
    # less stable close to erfc==1.  Since the Gaussian is symmetric
    # about 0, we can choose to flip the calculation around the origin
    # to improve our stability.  For example, if a is -inf, and b is
    # -100, then a naive application will return 0.  However, we can
    # just flip this around to be erfc(b)= erfc(100), which we can then
    # STABLY calculate with erfcx(100)...neat.  This leads to many
    # different cases, when either argument is inf or not.
    # Also there may appear to be some redundancy here, but it is also
    # worth noting that a less detailed application of erfcx can be
    # troublesome, as erfcx(-big numbers) = Inf, which wreaks havoc on
    # a lot of the calculations.  The cases below treat this when
    # necessary.
    # first check for problem cases
    # a problem case
    I = torch.isinf(a) & torch.isinf(b)
    if I.any():
        # check the sign
        I_sign = torch.eq(torch.sign(a),torch.sign(b)) & I
        # then this is integrating from inf to inf, for example.
        #logZhat = -inf
        #meanConst = inf
        #varConst = 0
        logZhat[I_sign] = torch.Tensor([-np.inf]).double().to(device=device)
        Zhat[I_sign] = torch.Tensor([0]).double().to(device=device)
        muHat[I_sign] = a[I_sign]
        sigmaHat[I_sign] = torch.Tensor([0]).double().to(device=device)
        entropy[I_sign] = torch.Tensor([-np.inf]).double().to(device=device)
        #logZhat = 0
        #meanConst = mu
        #varConst = 0

        I_sign_n = ~torch.eq(torch.sign(a),torch.sign(b)) & I
        logZhat[I_sign_n] = torch.Tensor([0]).double().to(device=device)
        Zhat[I_sign_n] = torch.Tensor([1]).double().to(device=device)
        muHat[I_sign_n] = mu[I_sign_n]
        sigmaHat[I_sign_n] = sigma[I_sign_n]
        entropy[I_sign_n] = 0.5*torch.log(2*pi[I_sign_n]*torch.exp(torch.Tensor([1])).double().to(device=device)*sigma[I_sign_n])
    # a problem case
    I_taken_care_of = I
    I = (a > b) & ~I_taken_care_of
    if I.any():
        # these bounds pointing the wrong way, so we return 0 by convention.
        #logZhat = -inf
        #meanConst = 0
        #varConst = 0
        logZhat[I] = torch.Tensor([-np.inf]).double().to(device=device)
        Zhat[I] = torch.zeroes_like(mu[I]).double().to(device=device)
        muHat[I] = mu[I]
        sigmaHAT[I] = torch.zeroes_like(mu[I]).double().to(device=device)
        entropy[I] = torch.Tensor([-np.inf]).double().to(device=device)

    # now real cases follow...
    I_taken_care_of = I | I_taken_care_of
    I = (torch.isinf(a)) & ~I_taken_care_of
    if I.any():
        # then we are integrating everything up to b
        # in infinite precision we just want normcdf(b), but that is not
        # numerically stable.
        # instead we use various erfcx.  erfcx scales very very well for small
        # probabilities (close to 0), but poorly for big probabilities (close
        # to 1).  So some trickery is required.
        I_b = (b >= 26) & I
        if I_b.any():
            # then this will be very close to 1... use this goofy expm1 log1p
            # to extend the range up to b==27... 27 std devs away from the
            # mean is really far, so hopefully that should be adequate.  I
            # haven't been able to get it past that, but it should not matter,
            # as it will just equal 1 thereafter.  Slight inaccuracy that
            # should not cause any trouble, but still no division by zero or
            # anything like that.
            # Note that this case is important, because after b=27, logZhat as
            # calculated in the other case will equal inf, not 0 as it should.
            # This case returns 0.
            logZhatOtherTail = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                               + torch.log(erfcx(b[I_b]))\
                               - b[I_b]**2
            logZhat[I_b] = torch.log1p(-torch.exp(logZhatOtherTail))

        I_b_n = (b < 26) & I
        if (I_b_n).any():
            # b is less than 26, so should be stable to calculate the moments
            # with a clean application of erfcx, which should work out to
            # an argument almost b==-inf.
            # this is the cleanest case, and the other moments are easy also...
            logZhat[I_b_n] = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                      + torch.log(erfcx(-b[I_b_n])) - b[I_b_n]**2

        # the mean/var calculations are insensitive to these calculations, as we do
        # not deal in the log space.  Since we have to exponentiate everything,
        # values will be numerically 0 or 1 at all the tails, so the mean/var will
        # not move.
        # note that the mean and variance are finally calculated below
        # we just calculate the constant here.
        meanConst[I] = -2./erfcx(-b[I])
        varConst[I] = -2./erfcx(-b[I])*(upperB[I] + mu[I])
        #   muHat = mu - (sqrt(sigma/(2*np.pi))*2)./erfcx(-b)
        #   sigmaHat = sigma + mu.^2 - muHat.^2 - (sqrt(sigma/(2*np.pi))*2)./erfcx(-b)*(upperB + mu)

    I_taken_care_of = I | I_taken_care_of
    I = torch.isinf(b) & ~I_taken_care_of
    if I.any():
        # then we are integrating from a up to Inf, which is just the opposite
        # of the above case.
        I_a = (a <= -26) & I
        a_erfcx = erfcx(a[I])
        if I_a.any():
            # then this will be very close to 1... use this goofy expm1 log1p
            # to extend the range up to a==27... 27 std devs away from the
            # mean is really far, so hopefully that should be adequate.  I
            # haven't been able to get it past that, but it should not matter,
            # as it will just equal 1 thereafter.  Slight inaccuracy that
            # should not cause any trouble, but still no division by zero or
            # anything like that.
            # Note that this case is important, because after a=27, logZhat as
            # calculated in the other case will equal inf, not 0 as it should.
            # This case returns 0.
            logZhatOtherTail = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                               + torch.log(erfcx(-a[I_a]))\
                               - a[I_a]**2
            logZhat[I_a] = torch.log1p(-torch.exp(logZhatOtherTail))

        I_a_n = (a > -26) & I
        if (I_a_n).any():
            # a is more than -26, so should be stable to calculate the moments
            # with a clean application of erfcx, which should work out to
            # almost inf.
            # this is the cleanest case, and the other moments are easy also...
            logZhat[I_a_n] = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                            + torch.log(erfcx(a[I_a_n]))\
                            - a[I_a_n]**2

        # the mean/var calculations are insensitive to these calculations, as we do
        # not deal in the log space.  Since we have to exponentiate everything,
        # values will be numerically 0 or 1 at all the tails, so the mean/var will
        # not move.
        meanConst[I] = 2./a_erfcx
        varConst[I] = 2./a_erfcx*(lowerB[I] + mu[I])
        #muHat = mu + (sqrt(sigma/(2*np.pi))*2)./erfcx(a)
        #sigmaHat = sigma + mu.^2 - muHat.^2 + (sqrt(sigma/(2*np.pi))*2)./erfcx(a)*(lowerB + mu)

    I_taken_care_of = I | I_taken_care_of
    # any other cases has bounds for which neither are inf
    I = ~I_taken_care_of
    # we have a range from a to b (neither inf), and we need some stable exponent
    if I.any():
        # calculations.
        I_eq = torch.eq(torch.sign(a),torch.sign(b)) & I
        if I_eq.any():
            # then we can exploit symmetry in this problem to make the
            # calculations stable for erfcx, that is, with positive arguments:
            # Zerfcx1 = 0.5*(exp(-b.^2)*erfcx(b) - exp(-a.^2)*erfcx(a))
            maxab = torch.max(torch.abs(a[I_eq]), torch.abs(b[I_eq]))
            minab = torch.min(torch.abs(a[I_eq]), torch.abs(b[I_eq]))
            logZhat[I_eq] = torch.log(torch.Tensor([0.5])).double().to(device=device) - minab**2 \
                      + torch.log( torch.abs( torch.exp(-(maxab**2-minab**2))*erfcx(maxab)\
                      - erfcx(minab)) )

            # now the mean and variance calculations
            # note here the use of the abs and signum functions for flipping the sign
            # of the arguments appropriately.  This uses the relationship
            # erfc(a) = 2 - erfc(-a).
            meanConst[I_eq] = 2*torch.sign(a[I_eq])*(1/((erfcx(abs(a[I_eq])) \
                                      - torch.exp(a[I_eq]**2-b[I_eq]**2)*erfcx(abs(b[I_eq]))))\
                                      - 1/((torch.exp(b[I_eq]**2-a[I_eq]**2)*erfcx(abs(a[I_eq]))\
                                      - erfcx(abs(b[I_eq])))))
            varConst[I_eq] =  2*torch.sign(a[I_eq])*((lowerB[I_eq]+mu[I_eq])/((erfcx(abs(a[I_eq]))\
                        - torch.exp(a[I_eq]**2-b[I_eq]**2)*erfcx(abs(b[I_eq]))))\
                        - (upperB[I_eq]+mu[I_eq])/((torch.exp(b[I_eq]**2-a[I_eq]**2)*erfcx(abs(a[I_eq]))\
                        - erfcx(abs(b[I_eq])))))

        I_n_eq = ~torch.eq(torch.sign(a),torch.sign(b)) & I
        if I_n_eq.any():
            # then the signs are different, which means b>a (upper>lower by definition), and b>=0, a<=0.
            # but we want to take the bigger one (larger magnitude) and make it positive, as that
            # is the numerically stable end of this tail.
            I_b_big_a = (torch.abs(b) >= torch.abs(a)) & I_n_eq
            if I_b_big_a.any():
                mask = (a >= -26) & I_b_big_a
                if mask.any():
                    # do things normally
                    logZhat[mask] = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                                    - a[mask]**2 + torch.log(erfcx(a[mask])\
                                                             - torch.exp(-(b[mask]**2\
                                                                           - a[mask]**2))*erfcx(b[mask]))

                    # now the mean and var
                    meanConst[mask] = 2*(1/((erfcx(a[mask])\
                                - torch.exp(a[mask]**2\
                                            -b[mask]**2)*erfcx(b[mask])))\
                                - 1/((torch.exp(b[mask]**2\
                                                -a[mask]**2)*erfcx(a[mask])\
                                      - erfcx(b[mask]))))
                    varConst[mask] = 2*((lowerB[mask]+mu[mask])/((erfcx(a[mask])\
                               - torch.exp(a[mask]**2-b[mask]**2)*erfcx(b[mask])))\
                               - (upperB[mask]+mu[mask])/((torch.exp(b[mask]**2-a[mask]**2)*erfcx(a[mask])\
                               - erfcx(b[mask]))))

                mask = (a < -26) & I_b_big_a
                if mask.any():
                    # a is too small and the calculation will be unstable, so
                    # we just put in something very close to 2 instead.
                    # Again this uses the relationship
                    # erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
                    # case makes sense.  This just says 2 - the right
                    # tail - the left tail.
                    logZhat[mask] = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                                    + torch.log( 2 - torch.exp(-b[mask]**2)*erfcx(b[mask])\
                                                 - torch.exp(-a[mask]**2)*erfcx(-a[mask]) )

                    # now the mean and var
                    meanConst[mask] = 2*(1/((erfcx(a[mask]) - torch.exp(a[mask]**2-b[mask]**2)*erfcx(b[mask])))\
                                - 1/(torch.exp(b[mask]**2)*2 - erfcx(b[mask])))
                    varConst[mask] = 2*((lowerB[mask]+mu[mask])/((erfcx(a[mask])\
                               - torch.exp(a[mask]**2-b[mask]**2)*erfcx(b[mask])))\
                               - (upperB[mask]+mu[mask])/(torch.exp(b[mask]**2)*2 - erfcx(b[mask])))

            I_b_less_a = (torch.abs(b) < torch.abs(a)) & I_n_eq
            if I_b_less_a.any():
                mask = (b <= 26) & I_b_less_a
                if mask.any():

                    # do things normally but mirrored across 0
                    logZhat[mask] = torch.log(torch.Tensor([0.5])).double().to(device=device) - b[mask]**2 + torch.log( erfcx(-b[mask])\
                              - torch.exp(-(a[mask]**2 - b[mask]**2))*erfcx(-a[mask]))

                    # now the mean and var
                    meanConst[mask] = -2*(1/((erfcx(-a[mask])\
                                - torch.exp(a[mask]**2-b[mask]**2)*erfcx(-b[mask])))\
                                - 1/((torch.exp(b[mask]**2-a[mask]**2)*erfcx(-a[mask])\
                                - erfcx(-b[mask]))))
                    varConst[mask] = -2*((lowerB[mask]+mu[mask])/((erfcx(-a[mask]) \
                               - torch.exp(a[mask]**2-b[mask]**2)*erfcx(-b[mask]))) \
                               - (upperB[mask]+mu[mask])/((torch.exp(b[mask]**2-a[mask]**2)*erfcx(-a[mask]) \
                               - erfcx(-b[mask]))))

                mask = (b > 26) & I_b_less_a
                if mask.any():

                    # b is too big and the calculation will be unstable, so
                    # we just put in something very close to 2 instead.
                    # Again this uses the relationship
                    # erfc(a) = 2 - erfc(-a). Since a<0 and b>0, this
                    # case makes sense. This just says 2 - the right
                    # tail - the left tail.
                    logZhat[mask] = torch.log(torch.Tensor([0.5])).double().to(device=device)\
                              + torch.log( 2 - torch.exp(-a[mask]**2)*erfcx(-a[mask])\
                              - torch.exp(-b[mask]**2)*erfcx(b[mask]) )

                    # now the mean and var
                    meanConst[mask] = -2*(1/(erfcx(-a[mask]) - torch.exp(a[mask]**2)*2)\
                                - 1/(torch.exp(b[mask]**2-a[mask]**2)*erfcx(-a[mask]) - erfcx(-b[mask])))
                    varConst = -2*((lowerB[mask] + mu[mask])/(erfcx(-a[mask])\
                               - torch.exp(a[mask]**2)*2)\
                               - (upperB[mask] + mu[mask])/(torch.exp(b[mask]**2-a[mask]**2)*erfcx(-a[mask])\
                               - erfcx(-b[mask])))

            # the above four cases (diff signs x stable/unstable) can be
            # collapsed into two cases by tracking the sign of the maxab
            # and sign of the minab (the min and max of abs(a) and
            # abs(b)), but that is a bit less clear, so we
            # leave it fleshed out above.



    """
    # finally, calculate the returned values
    """
    # logZhat is already calculated, as are meanConst and varConst.
    # no numerical precision in Zhat
    Zhat = torch.exp(logZhat)
    # make the mean
    muHat = mu + meanConst*torch.sqrt(sigma/(2*pi))
    # make the var
    sigmaHat = sigma + varConst*torch.sqrt(sigma/(2*pi)) + mu**2 - muHat**2
    # make entropy
    entropy = 0.5*((meanConst*torch.sqrt(sigma/(2*pi)))**2
                + sigmaHat - sigma)/sigma\
              + logZhat + torch.log(torch.sqrt(2*pi*torch.exp(torch.Tensor([1]).double().to(device=device))))\
              + torch.log(torch.sqrt(sigma))
    return logZhat.float(), Zhat.float(), muHat.float(), sigmaHat.float(), entropy.float()
