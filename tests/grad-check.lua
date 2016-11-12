local autograd = require 'autograd'
local finiteDifferenceEpsilonDefault = 1e-11
local maximumErrorDefault = 1e-1

-- First order finite difference approximation of f'(x) where f takes either
-- numerical or Tensor input and outputs a number.
-- In case of numerical input, this function returns a numerical derivative.
-- In case of Tensor input, this function returns a Tensor gradient.
-- https://en.wikipedia.org/wiki/Finite_difference_method
function gradFromFiniteDifference(f, x, epsilon)
    if epsilon == nil then
        epsilon = finiteDifferenceEpsilonDefault
    end

    if torch.isTensor(x) then
        local D = x:size(1)
        local res = torch.Tensor(D)
        local fx = f(x)
        for d = 1, D do
            local epsilonTensor = torch.zeros(D)
            epsilonTensor[d] = epsilon
            res[d] = (f(x + epsilonTensor) - fx) / epsilon
        end
        return res
    else
        local fDiff = f(x + epsilon) - f(x)
        return fDiff / epsilon
    end
end

-- Gradient or derivative of f at x, from Torch Autograd.
function gradFromTorchAutograd(f, x)
    local df = autograd(f)
    local dfdx, fx = df(x)

    return dfdx
end

function norm(x)
    if torch.isTensor(x) then
        return torch.norm(x)
    else
        return torch.abs(x)
    end
end

-- Checks Torch Autograd gradient of f at x by comparing it with approximate
-- gradient from the finite difference method.
-- Returns true if within maximumError, otherwise false.
function gradCheck(f, x, maximumError, finiteDifferenceEpsilon)
    if maximumError == nil then
        maximumError = maximumErrorDefault
    end

    local dfdxFD = gradFromFiniteDifference(f, x, finiteDifferenceEpsilon)
    local dfdxTA = gradFromTorchAutograd(f, x)
    local norm = norm(dfdxFD - dfdxTA)
    if norm < maximumError then
        return true
    else
        print('\n-----\n')
        print('Wrong gradient:')
        print('Input (x):')
        print(x)
        print('Gradient from Torch Autograd:')
        print(dfdxTA)
        print('Gradient from Finite Differencing:')
        print(dfdxFD)
        print(string.format('Error (norm of difference): %f', norm))
        print(string.format('Maximum error:              %f', maximumError))
        print('\n-----\n')
        return false
    end
end
