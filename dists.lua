dists = {}

cephes = require 'cephes'
distributions = require 'distributions'
overload = require 'autograd.overload'

-- Derivatives of beta function: https://en.wikipedia.org/wiki/Beta_function
-- Used in distribution pdfs such as continuousminmax in compile.lua.
overload.module("cephes", cephes, function(module)
   module.gradient("beta", {
      function(g, ans, a, b)
          return g * cephes.betagrad(a, b)
      end,
      function(g, ans, a, b)
          return g * cephes.betagrad(b, a)
      end
   })
end)

-- Derivatives of cephes.gamma: https://en.wikipedia.org/wiki/Gamma_function
-- Currently not used.
overload.module("cephes", cephes, function(module)
   module.gradient("gamma", {
      function(g, ans, x)
          return torch.cmul(g, torch.cmul(cephes.gamma(x), cephes.polygamma(0, x)))
      end
   })
end)

-- Derivatives of cephes.lgam:
-- https://en.wikipedia.org/wiki/Gamma_function#The_log-gamma_function
-- Currently not used but can be used in dirichletLogpdf.
overload.module("cephes", cephes, function(module)
   module.gradient("lgam", {
      function(g, ans, x)
          if type(x) == 'number' then
              return g * cephes.polygamma(0, x)
          else
              return torch.cmul(g, cephes.polygamma(0, x))
          end
        end
   })
end)

-- Copied from https://github.com/deepmind/torch-distributions/blob/master/distributions/multivariateGaussian.lua
-- Changed Tensor creation to adapt to the type of the argument Tensor because
-- original code is not compatible with Tensors of different types.
function dists.mvnLogpdf(x, mu, sigma, options)
    options = options or {}
    x = x:clone()
    mu = mu:clone()
    sigma = sigma:clone()

    -- If any input is vectorised, we return a vector. Otherwise remember that we should return scalar.
    local scalarResult = (x:dim() == 1) and (mu:dim() == 1)

    -- Now make our inputs all vectors, for simplicity
    if x:dim() == 1 then
        x = x:view(1, x:nElement())
    end
    if mu:dim() == 1 then
        mu = mu:view(1, mu:nElement())
    end

    -- Expand any 1-row inputs so that we have matching sizes
    local nResults
    if x:size(1) == 1 and mu:size(1) ~= 1 then
        nResults = mu:size(1)
        x = x:expand(nResults, x:size(2))
    elseif x:size(1) ~= 1 and mu:size(1) == 1 then
        nResults = x:size(1)
        mu = mu:expand(nResults, mu:size(2))
    else
        if x:size(1) ~= mu:size(1) then
            error("x and mu should have the same number of rows")
        end
        nResults = x:size(1)
    end

    x = x:clone():add(-1, mu)

    local logdet
    local transformed
    local decomposed

    -- For a diagonal covariance matrix, we allow passing a vector of the diagonal entries
    if sigma:dim() == 1 then
        local D = sigma:size(1)
        decomposed = sigma:clone()
        if not options.cholesky then
            decomposed:sqrt()
        end
        transformed = torch.cdiv(x, decomposed:view(1, D):expand(nResults, D))
        logdet = decomposed:log():sum()
    else
        if not options.cholesky then
            decomposed = torch.potrf(sigma):triu() -- TODO remove triu as torch will be fixed
        else
            decomposed = sigma
        end
        transformed = torch.gesv(x:t(), decomposed:t()):t()
        logdet = decomposed:diag():log():sum()
    end
    transformed:apply(function(a) return distributions.norm.logpdf(a, 0, 1) end)
    local result = transformed:sum(2) - logdet -- by independence
    if scalarResult then
        return result[1][1]
    else
        return result
    end
end

-- Custom derivatives of mvnLogpdf
overload.module("dists", dists, function(module)
    module.gradient("mvnLogpdf", {
        function(g, ans, x, mu, sigma, options)
            return g * torch.gesv(torch.view(mu - x, -1, 1), sigma):view(-1)
        end,
        function(g, ans, x, mu, sigma, options)
            return g * torch.gesv(torch.view(x - mu, -1, 1), sigma)
        end,
        function(g, ans, x, mu, sigma, options)
            local df_dmu = torch.gesv(torch.view(x - mu, -1, 1), sigma):view(-1)
            return g * -0.5 * (torch.inverse(sigma) - torch.ger(df_dmu, df_dmu))
        end,
        function(g, ans, x, mu, sigma, options)
            return options
        end
    })
end)

-- Custom derivatives of distributions.norm.cdf
-- Used in foldedNormalDiscreteLogpdf
overload.module("distributions.norm", distributions.norm, function(module)
   module.gradient("cdf", {
      function(g, ans, x, mu, sigma)
          return g * distributions.norm.pdf(x, mu, sigma)
      end,
      function(g, ans, x, mu, sigma)
          return -g * distributions.norm.pdf(x, mu, sigma)
      end,
      function(g, ans, x, mu, sigma)
          return -g * ((x - mu) / sigma) * distributions.norm.pdf(x, mu, sigma)
      end
   })
end)

-- Discretised version of the folded normal distribution
-- https://en.wikipedia.org/wiki/Folded_normal_distribution
-- Has range: nonnegative integers {0, 1, 2, ...}.
function dists.foldedNormalDiscreteLogpdf(x, mean, sd)
    local res = 0
    if (x < 0) or (x ~= torch.round(x)) then
        res = torch.log(epsilon)
    else
        if x == 0 then
            res = torch.log(distributions.norm.cdf(0.5, mean, sd) - distributions.norm.cdf(-0.5, mean, sd))
        else
            res = torch.log(
                distributions.norm.cdf(x + 0.5, mean, sd) -  distributions.norm.cdf(x - 0.5, mean, sd) +
                distributions.norm.cdf(-x + 0.5, mean, sd) - distributions.norm.cdf(-x - 0.5, mean, sd)
            )
        end
    end

    -- TODO: fix this hack to prevent numerical issues
    if res == torch.log(0) or (type(res) == 'table' and res.value == torch.log(0)) then
        res = -20
    end

    return res
end

-- use this instead of https://github.com/deepmind/torch-distributions/blob/master/distributions/dirichlet.lua#L57
-- to avoid in-place calculations which are currently not supported by Torch Autograd
-- Note that currently, lgam causes malloc errors
-- https://github.com/deepmind/torch-cephes/blob/f8a3a110c369bd1f65253b35436f3037458b4264/cephes/cprob/gamma.c
-- lgam still causes problems, even when writing custom derivatives
function dists.dirichletLogpdf(x, alpha)
    return torch.log(x) * (alpha - 1) - torch.sum(cephes.lgam(alpha)) + cephes.lgam(torch.sum(alpha))
end

-- Custom derivatives for dirichletLogPdf
overload.module("dists", dists, function(module)
    module.gradient("dirichletLogpdf", {
        function(g, ans, x, alpha)
            return g * torch.cmul(alpha - 1, torch.cinv(x))
        end,
        function(g, ans, x, alpha)
            return g * (cephes.psi(torch.sum(alpha)) - cephes.psi(alpha) + torch.log(x))
        end
    })
end)
