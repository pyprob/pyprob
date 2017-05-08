--
-- INFERENCE COMPILATION
-- Utility functions
--
-- Tuan-Anh Le, Atilim Gunes Baydin
-- tuananh@robots.ox.ac.uk; gunes@robots.ox.ac.uk
-- University of Oxford
-- May 2016 -- March 2017
--

versionString = '0.8.3'
epsilon = 1e-5

torch.manualSeed(1)

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'gnuplot'
zmq = require 'lzmq'
mp = require 'MessagePack'
colors = require 'ansicolors'
lfs = require 'lfs'
hash = require 'hash'

local has_socket, socket = pcall(require, "socket")

--  Sleep for a number of milliseconds
if has_socket then
    function s_sleep(msecs)
        socket.sleep(msecs / 1000)
    end
else
    function s_sleep(msecs)
        os.execute("sleep " .. tostring(msecs / 1000))
    end
end


if opt then
    local logFile = (io.type(opt.log) == 'file' and file) or io.open(opt.log, 'w')
    function printLog(l)
        local s = tostring(l or '')
        io.write(colors(s) .. '\n')
        local fs = string.gsub(s, '(%%{(.-)})', function(_, str) return '' end)
        logFile:write(fs .. '\n')
        logFile:flush()
    end

    if opt.cuda then
        require 'cutorch'
        require 'cunn'
        require 'cudnn'
        cudnn.fastest = true
        cudnn.benchmark = false
        cutorch.setDevice(opt.device)
        cutorch.manualSeed(1)
        Tensor = torch.CudaTensor
    else
        require 'nn'
        Tensor = torch.Tensor
    end

    function moveToCuda(orig)
        local copy = orig
        if opt.cuda then
            if type(orig) == 'table' then
                if torch.isTypeOf(orig, nn.Module) then
                    copy = orig:cuda()
                    -- cudnn.convert(copy, cudnn)
                else
                    copy = {}
                    for orig_key, orig_value in next, orig, nil do
                        copy[moveToCuda(orig_key)] = moveToCuda(orig_value)
                    end
                    setmetatable(copy, moveToCuda(getmetatable(orig)))
                end
            elseif torch.isTensor(orig) then
                copy = orig:cuda()
            end
        end
        return copy
    end

    function moveToHost(orig)
        local copy = orig
        if opt.cuda then
            if type(orig) == 'table' then
                if torch.isTypeOf(orig, nn.Module) then
                    copy = orig:double()
                else
                    copy = {}
                    for orig_key, orig_value in next, orig, nil do
                        copy[moveToHost(orig_key)] = moveToHost(orig_value)
                    end
                    setmetatable(copy, moveToHost(getmetatable(orig)))
                end
            elseif torch.isTensor(orig) then
                copy = orig:double()
            end
        end
        return copy
    end
end

io.write('\n')
autograd = require 'autograd'
io.write(string.char(27) .. '[1A                     \r')
io.flush()

require 'rnn'
autograd.optimize(false)

function standardize(t)
    local mean = torch.mean(t)
    local sd = torch.std(t)
    t:add(-mean)
    t:div(sd + epsilon)
    return t
end

function formatThousand(v, decimals)
	local s = string.format("%d", math.floor(v))
	local pos = string.len(s) % 3
	if pos == 0 then pos = 3 end
    local ret = string.sub(s, 1, pos)
		.. string.gsub(string.sub(s, pos+1), "(...)", ",%1")
    if decimals then
        ret = ret .. string.sub(string.format("%.2f", v - math.floor(v)), 2)
    end
    return ret
end

spinner = {'│\r', '/\r', '─\r', '\\\r'}
spinneri = 1
function spin()
    io.write(spinner[spinneri])
    io.flush()
    spinneri = spinneri + 1
    if spinneri > 4 then spinneri = 1 end
end

function fileStartingWith(name, n)
    local pfile = io.popen('ls -a ' .. name .. '*')
    local latest = nil
    local count = 0
    for filename in pfile:lines() do
        latest = filename
        count = count + 1
        if count == n then break end
    end
    return latest
end

function daysHoursMinsSecs(secs)
    local t = os.date('!*t', secs)
    return string.format('%id %.2d:%.2d:%.2d', t.yday - 1, t.hour, t.min, t.sec)
end

function tableLength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

function tableInvert(t)
   local s = {}
   for k, v in pairs(t) do
     s[v] = k
   end
   return s
end

local resampleMatrixCache = {}
function resampleMatrix(sourceDim, targetDim)
    resampleMatrixCache[sourceDim] = resampleMatrixCache[sourceDim] or {}
    local m = resampleMatrixCache[sourceDim][targetDim]
    if not m then
        -- print('not in dict')
        m = {}
        for i = 1, targetDim do
            mrow = torch.zeros(sourceDim)
            local tlow = (i - 1) * (sourceDim / targetDim)
            local thigh = i * (sourceDim / targetDim)
            -- print('tlow ' .. tlow .. ' thigh ' .. thigh)
            for j = 1, sourceDim do
                local slow = j - 1
                local shigh = j
                -- print('slow ' .. slow .. ' shigh ' .. shigh)
                if slow >= tlow and shigh <= thigh then
                    mrow[j] = 1
                -- elseif tlow >= shigh or thigh <= slow then
                --     mrow[j] = 0
                else
                    if thigh > slow and thigh < shigh then
                        mrow[j] = thigh - slow
                    end
                    if tlow < shigh and tlow > slow then
                        mrow[j] = shigh - tlow
                    end
                end
            end
            mrow = mrow / mrow:sum()
            m[i] = mrow
        end
        m = nn.JoinTable(1):forward(m):view(targetDim, -1)
        resampleMatrixCache[sourceDim][targetDim] = m
    end
    return m
end

function loadArtifact(file)
    assert(io.open(file, "r"), 'Cannot read artifact file '..file)
    local artifact = torch.load(file)
    if artifact.codeVersion ~= versionString then
        printLog('%{bright red}Warning: Loaded artifact was saved with another version of code.\n')
    end
    local fileSize = lfs.attributes (file, 'size')
    local iterPerSec = artifact.totalIterations / artifact.totalTrainingTime
    local tracesPerSec = artifact.totalTraces / artifact.totalTrainingTime
    local tracesPerIter = artifact.totalTraces / artifact.totalIterations
    local lossChange = artifact.validLossFinal - artifact.validLossInitial
    local lossChangePerSec = lossChange / artifact.totalTrainingTime
    local lossChangePerIter = lossChange / artifact.totalIterations
    local lossChangePerTrace = lossChange / artifact.totalTraces
    local addresses = ''
    for address, _ in pairs(artifact.oneHotDict['address']) do addresses = addresses .. ' ' .. address end
    local instances = ''
    for instance, _ in pairs(artifact.oneHotDict['instance']) do instances = instances .. ' ' .. instance end
    local proposalTypes = ''
    for proposalType, _ in pairs(artifact.oneHotDict['proposalType']) do proposalTypes = proposalTypes .. ' ' .. proposalType end
    local info =
        string.format('File name             : %s\n', file) ..
        string.format('File size             : %s bytes\n', formatThousand(fileSize)) ..
        string.format('Created               : %s\n', artifact.created) ..
        string.format('Last modified         : %s\n', artifact.modified) ..
        string.format('Code version          : %s\n', artifact.codeVersion) ..
        string.format('Cuda                  : %s\n', tostring(artifact.cuda)) ..
        string.format('%%{bright cyan}Trainable params      : %s\n', formatThousand(artifact.trainableParams)) ..
        string.format('%%{bright yellow}Total training time   : %s\n', daysHoursMinsSecs(artifact.totalTrainingTime)) ..
        string.format('%%{dim yellow}Updates to file       : %s\n', formatThousand(artifact.updates)) ..
        string.format('Iterations            : %s\n', formatThousand(artifact.totalIterations)) ..
        string.format('Iterations / s        : %s\n', formatThousand(iterPerSec, true)) ..
        string.format('%%{reset}%%{bright yellow}Total training traces : %s\n', formatThousand(artifact.totalTraces)) ..
        string.format('%%{dim yellow}Traces / s            : %s\n', formatThousand(tracesPerSec, true)) ..
        string.format('Traces / iteration    : %s\n', formatThousand(tracesPerIter, true)) ..
        string.format('%%{dim green}Initial loss          : %+e\n', artifact.validLossInitial) ..
        string.format('%%{reset}%%{bright green}Final   loss          : %+e\n', artifact.validLossFinal) ..
        string.format('%%{dim green}Loss change / s       : %+e\n', lossChangePerSec) ..
        string.format('Loss change / iter.   : %+e\n', lossChangePerIter) ..
        string.format('Loss change / trace   : %+e\n', lossChangePerTrace) ..
        string.format('Validation set size   : %s\n', artifact.validSize) ..
        string.format('%%{dim cyan}Observe embedding     : %s\n', artifact.obsEmb) ..
        string.format('Observe emb. dim      : %i\n', artifact.obsEmbDim) ..
        string.format('Sample embedding      : %s\n', artifact.smpEmb) ..
        string.format('Sample emb. dim       : %i\n', artifact.smpEmbDim) ..
        string.format('LSTM dim              : %i\n', artifact.lstmDim) ..
        string.format('LSTM depth            : %i\n', artifact.lstmDepth) ..
        string.format('Softmax dim           : %i\n', artifact.softMaxDim) ..
        string.format('Softmax boost         : %i\n', artifact.softMaxBoost) ..
        string.format('Dirichlet dim         : %i\n', artifact.dirichletDim) ..
        string.format('%%{yellow}Addresses             :%s\n', addresses) ..
        string.format('Instances             :%s\n', instances) ..
        string.format('Proposal types        :%s\n', proposalTypes)
    return artifact, info, artifact.trainHistoryTrace, artifact.trainHistoryLoss, artifact.validHistoryTrace, artifact.validHistoryLoss
end
