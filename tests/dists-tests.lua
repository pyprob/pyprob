require 'dists'
require 'tests/grad-check'
local distsTests = torch.TestSuite()
local tester = torch.Tester()
local epsilon = 1e-3

function distsTests.testFoldedNormalDiscreteLogpdf()
    local xs = {10, 9, 8, 5}
    local means = {2, 3, 1, 3}
    local sds = {1, 2, 3, 4}
    local probs = {-31.073840089541154, -6.03114450666107, -4.560889600253781, -2.288211296784738}

    for i = 1, #xs do
        x = xs[i]
        mean = means[i]
        sd = sds[i]
        probGround = probs[i]
        prob = dists.foldedNormalDiscreteLogpdf(x, mean, sd)

        tester:assert((prob > probGround - epsilon) and (prob < probGround + epsilon), 'incorrect value for folded normal discrete logpdf')
    end
end

function distsTests.testFoldedNormalDiscreteLogpdfGrad()
    local testFoldedNormalDiscreteLogpdfFunction = function(params)
        local mean = torch.abs(params[1])
        local sd = torch.abs(params[2])
        return dists.foldedNormalDiscreteLogpdf(1, mean, sd)
    end

    for i = 1, 10 do
        local mean = torch.Tensor(1):normal():abs()
        local sd = torch.Tensor(1):normal():abs()

        tester:assert(gradCheck(testFoldedNormalDiscreteLogpdfFunction, torch.cat{mean, sd}), 'incorrect gradients for folded normal discrete logpdf')
    end
end

function distsTests.testDirichletLogPdfGrad()
    local testDirichletLogpdfFunction = function(params)
        local x = torch.abs(params[{{1, 3}}])
        x = x / torch.sum(x)
        local alpha = torch.abs(params[{{4, 6}}])
        return dists.dirichletLogpdf(x, alpha)
    end

    for i = 1, 10 do
        local x = torch.Tensor(3):normal():abs()
        x = x / torch.sum(x)
        local alpha = torch.Tensor(3):normal():abs()

        tester:assert(gradCheck(testDirichletLogpdfFunction, torch.cat(x, alpha)), 'incorrect gradients for dirichlet logpdf')
    end
end

function distsTests.testNormCdfGrad()
    local normCdf = function(params)
        local x = params[1]
        local mean = params[2]
        local sd = torch.abs(params[3])
        return distributions.norm.cdf(x, mean, sd)
    end

    for i = 1, 10 do
        local x = torch.Tensor(1):normal()
        local mean = torch.Tensor(1):normal()
        local sd = torch.Tensor(1):normal():abs()

        tester:assert(gradCheck(normCdf, torch.cat({x, mean, sd})), 'incorrect gradients for beta function')
    end
end

function distsTests.testBetaGrad()
    local betaFunction = function(params)
        local alpha = torch.abs(params[1])
        local beta = torch.abs(params[2])
        return cephes.beta(alpha, beta)
    end

    for i = 1, 10 do
        local alpha = torch.Tensor(1):normal():abs()
        local beta = torch.Tensor(1):normal():abs()

        tester:assert(gradCheck(betaFunction, torch.cat(alpha, beta)), 'incorrect gradients for beta function')
    end
end

function distsTests.testGammaGrad()
    local gammaFunction = function(x)
        local x = torch.abs(x)
        return cephes.gamma(x)[1]
    end

    for i = 1, 10 do
        local x = torch.Tensor(1):normal():abs()
        tester:assert(gradCheck(gammaFunction, x), 'incorrect gradients for gamma function')
    end
end

function distsTests.testLogGammaGrad()
    local logGammaFunction = function(x)
        local x = torch.abs(x)
        return cephes.lgam(x)
    end

    for i = 1, 10 do
        local x = torch.Tensor(1):normal():abs()
        tester:assert(gradCheck(logGammaFunction, x[1]), 'incorrect gradients for log gamma function')
    end
end

function distsTests.testMvnLogpdfGrad()
    local mvnLogpdfFunction = function(params)
        local x = params[{{1, 2}}]
        local mu = params[{{3, 4}}]
        local preCov = torch.view(params[{{5, 8}}], 2, 2)
        local cov = preCov + torch.t(preCov) + 2 * torch.eye(2)
        return dists.mvnLogpdf(x, mu, cov)
    end

    for i = 1, 10 do
        local x = torch.Tensor(2):normal()
        local mu = torch.Tensor(2):normal()
        local preCov = torch.rand(4)

        tester:assert(gradCheck(mvnLogpdfFunction, torch.cat({x, mu, preCov})), 'incorrect gradients for mvn log pdf')
    end
end

tester:add(distsTests)
tester:run()
