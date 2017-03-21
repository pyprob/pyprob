--
-- INFERENCE COMPILATION
-- Compilation mode
--
-- Tuan-Anh Le, Atilim Gunes Baydin
-- tuananh@robots.ox.ac.uk; gunes@robots.ox.ac.uk
-- University of Oxford
-- May 2016 -- March 2017
--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Oxford Inference Compilation')
cmd:text('Compilation mode')
cmd:text()
cmd:text('Options:')

cmd:option('--help', false, 'display this help')
cmd:option('--version', false, 'display version information')

-- training options
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--log', './artifacts/compile-log', 'file for logging')
cmd:option('--batchSize', 128, 'training batch size')
cmd:option('--validSize', 256, 'validation set size')
cmd:option('--validInterval', 500, 'validation interval (traces)')
cmd:option('--validNew', false, 'renew and replace the validation set, if resuming an existing artifact')
cmd:option('--learningRate', 0.0001, 'learning rate')
cmd:option('--learningRateDecay', 1e-7, 'learning rate decay (only with --method sgd)')
cmd:option('--weightDecay', 0.0005, 'weight decay')
cmd:option('--momentum', 0.9, 'momentum (only with --method sgd)')
cmd:option('--gradClip', -1, 'cutoff norm for gradient clipping')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--method', 'adam', 'optimization method (adam, sgd)')

-- architectural options
cmd:option('--obsEmb', 'fc', 'observation embedding type (fc, lenet, cnn6, cnn7, vgg)')
cmd:option('--obsEmbDim', 512, 'observation embedding dimension')
cmd:option('--smpEmb', 'fc', 'sample embedding type (fc)')
cmd:option('--smpEmbDim', 1, 'sample embedding dimension')
cmd:option('--lstmDim', 512, 'number of hidden units in the lstm')
cmd:option('--lstmDepth', 1, 'number of stacked lstms')
 -- TO DO: Implement softmax resampling. The size of softMaxDim affects inference performance, because we are pushing through ZMQ softMaxDim number of units even when only the first few of these are used.
cmd:option('--softMaxDim', 256, '')
cmd:option('--softMaxBoost', 20, '')
cmd:option('--dirichletDim', 256, '')
cmd:option('--oneHotDim', 64, '')
cmd:option('--noStandardize', false, 'no standardization of the neural net input')
cmd:option('--obsSmooth', false, 'smooth the observe embedding by a 5x5, std=0.5, gaussian kernel')

-- communication options
cmd:option('--resume', '', 'resume training of an existing artifact')
cmd:option('--resumeLatest', false, 'resume the training of the latest artifact file starting with the name given with --artifact')
cmd:option('--server', '127.0.0.1:5555', 'address and port of the Anglican server')
cmd:option('--artifact', './artifacts/compile-artifact', 'file to save the artifact (will be appended by a timestamp of the form -yymmdd-hhmmss)')
cmd:option('--keepArtifacts', false, 'keep all previously best artifacts during training, do not overwrite')

cmd:text()

opt = cmd:parse(arg or {})

if opt.help then
    cmd:help()
    do return end
end

function getTimeStamp()
    return os.date('-%y%m%d-%H%M%S')
end

local timeStamp = getTimeStamp()
opt.log = opt.log .. timeStamp
require 'util'
if opt.version then
    print('Oxford Inference Compilation')
    print('Compilation mode')
    print(versionString)
    do return end
end
require 'dists'
if opt.resumeLatest then
    opt.resume = fileStartingWith(opt.artifact, -1)
end
local artifactFile = opt.artifact .. timeStamp

local artifact = {}
local artifactInfo = {}

-- ZMQ COMMUNICATIONS
local zmqContext = {}
local zmqSocket = {}
function zmqSendRequest(comm)
    zmqContext = zmq.init(1)
    zmqSocket = zmqContext:socket(zmq.REQ)
    zmqSocket:connect('tcp://'..opt.server)

    local msgreq = mp.pack(comm)
    zmqSocket:send(msgreq)
end
function zmqReceiveReply()
    local msgrep = zmqSocket:recv()

    zmqSocket:close()
    zmqContext:term()

    local reply = {}
    for _, v in mp.unpacker(msgrep) do
        reply = v
    end
    return reply
end

-- zmqSendRequest({command = 'newbatch', param = 3})
-- data = zmqReceiveReply()
--
-- zmqSendRequest({command = 'newbatch', param = 3})
-- data = zmqReceiveReply()
--
-- print(data)
-- do return end

-- Process the received training data into training traces
-- The first call to this initializes the oneHotDict for different address, instance, and proposalType values
-- Thus we should ensure that the first training data request (the validation batch) is sufficiently large and representative
function getBatchTraces(data)
    local addressSet = {}
    local instanceSet = {}
    local proposalTypeSet = {}

    local observes = {}
    local samples = {}
    for m, v in pairs(data) do

        local obsshape = v['observes']['shape']
        local obsdata = v['observes']['data']

        local s = torch.Storage(obsdata)

        -- BIG HACK TO WORK AROUND THE FACT THAT conv2 is not implemented in cutorch:
        -- https://github.com/torch/cutorch/issues/70
        if artifact.obsSmooth then
            observes[m] = torch.Tensor(torch.LongStorage(obsshape))
            observes[m]:storage():copy(s)

            -- We can also use nn.SpatialContrastiveNormalization here
            if not(artifact.noStandardize) then
                observes[m] = standardize(observes[m])
            end

            require 'image'
            artifact.obsSmoothKernel = image.gaussian(5, 0.5, nil, true)
            observes[m] = moveToCuda(image.convolve(observes[m], artifact.obsSmoothKernel))
        else
            observes[m] = Tensor(torch.LongStorage(obsshape))
            observes[m]:storage():copy(s)

            -- We can also use nn.SpatialContrastiveNormalization here
            if not(artifact.noStandardize) then
                observes[m] = standardize(observes[m])
            end
        end

        samples[m] = samples[m] or {}
        for timeStep, vv in pairs(v['samples']) do
            samples[m][timeStep] = samples[m][timeStep] or {}
            local address = vv['sample-address']
            local instance = vv['sample-instance']
            local proposalType = vv['proposal-name']

            samples[m][timeStep].value = nil
            if type(vv['value']) == 'boolean' then
                -- if the received value is from a `flip` random variable
                if vv['value'] then
                    samples[m][timeStep].value = Tensor{1}
                else
                    samples[m][timeStep].value = Tensor{0}
                end
            elseif type(vv['value']) == 'number' then
                -- if the received value is just a number
                samples[m][timeStep].value = Tensor({vv['value']})
            elseif type(vv['value']) == 'table' then
                -- if the received value is a table
                samples[m][timeStep].value = Tensor(vv['value'])
            else
                samples[m][timeStep].value = vv['value']
            end
            -- Flatten the sample value so that it is always a 1-d tensor
            samples[m][timeStep].value = samples[m][timeStep].value:view(samples[m][timeStep].value:nElement())

            samples[m][timeStep].address = address
            samples[m][timeStep].instance = instance
            samples[m][timeStep].proposalType = proposalType

            if proposalType == 'categorical' then
                samples[m][timeStep].numCategories = vv['proposal-extra-params'][1]
                samples[m][timeStep].categories = vv['proposal-extra-params'][2]
            elseif proposalType == 'continuousminmax' then
                samples[m][timeStep].min = vv['proposal-extra-params'][1]
                samples[m][timeStep].max = vv['proposal-extra-params'][2]
            elseif proposalType == 'dirichlet' then
                samples[m][timeStep].alphaDim = vv['proposal-extra-params'][1]
            elseif proposalType == 'discreteminmax' then
                samples[m][timeStep].min = vv['proposal-extra-params'][1]
                samples[m][timeStep].max = vv['proposal-extra-params'][2]
            elseif proposalType == 'mvn' then
                samples[m][timeStep].dim = vv['proposal-extra-params'][1]
            elseif proposalType == 'mvnmeanvar' then
                samples[m][timeStep].dim = vv['proposal-extra-params'][1]
            elseif proposalType == 'mvnmeanvars' then
                samples[m][timeStep].dim = vv['proposal-extra-params'][1]
            end

            -- Update the oneHotDict as needed
            if not artifact.oneHotDict then
                artifact.oneHotDict = {}
                artifact.oneHotDict['address'] = {}
                artifact.oneHotDict['instance'] = {}
                artifact.oneHotDict['proposalType'] = {}
                artifact.oneHotDict['numAddress'] = 0
                artifact.oneHotDict['numInstance'] = 0
                artifact.oneHotDict['numProposalType'] = 0
            end
            if not artifact.oneHotDict['address'][address] then
                printLog('%{bright magenta}Polymorphing artifact, new address         : ' .. address)
                local i = artifact.oneHotDict['numAddress'] + 1
                local t = Tensor(artifact.oneHotDimAddress)
                t:fill(0):narrow(1, i, 1):fill(1)
                artifact.oneHotDict['address'][address] = t
                artifact.oneHotDict['numAddress'] = i
            end
            if not artifact.oneHotDict['instance'][instance] then
                printLog('%{bright magenta}Polymorphing artifact, new instance        : ' .. instance)
                local i = artifact.oneHotDict['numInstance'] + 1
                local t = Tensor(artifact.oneHotDimInstance)
                t:fill(0):narrow(1, i, 1):fill(1)
                artifact.oneHotDict['instance'][instance] = t
                artifact.oneHotDict['numInstance'] = i
            end
            if not artifact.oneHotDict['proposalType'][proposalType] then
                printLog('%{bright magenta}Polymorphing artifact, new proposal type   : ' .. proposalType)
                local i = artifact.oneHotDict['numProposalType'] + 1
                local t = Tensor(artifact.oneHotDimProposalType)
                t:fill(0):narrow(1, i, 1):fill(1)
                artifact.oneHotDict['proposalType'][proposalType] = t
                artifact.oneHotDict['numProposalType'] = i
            end
        end
    end

    local traces = {}
    for batchIndex = 1, #observes do
        local o = observes[batchIndex]
        local s = samples[batchIndex]
        local timeSteps = #s


        -- Do the trace hashing for subsequent splitting of batches into subbatches of matching hash values (that is, all traces in a subbatch have the same number of time steps and the same sequence of addresses and instances.)
        local traceHash = {}
        for i = 1, timeSteps do
            traceHash[#traceHash + 1] = moveToHost(artifact.oneHotDict['address'][s[i].address])
            traceHash[#traceHash + 1] = moveToHost(artifact.oneHotDict['instance'][s[i].instance])
        end
        traceHash = hash.hash(nn.JoinTable(1):forward(traceHash))

        local trace = {}
        trace['timeSteps'] = timeSteps
        trace['observes'] = o
        trace['samples'] = s
        trace['traceHash'] = traceHash

        traces[#traces + 1] = trace
    end
    return traces
end



function getSubBatches(batchTraces)
    local subBatches = {}

    for i = 1, #batchTraces do
        local traceHash = batchTraces[i].traceHash
        subBatches[traceHash] = subBatches[traceHash] or {}
        subBatches[traceHash][#(subBatches[traceHash]) + 1] = batchTraces[i]
    end

    return subBatches
end

-- ask for a new batch of size n from Anglican
function requestBatch(n)
    -- signal Anglican to generate a new batch
    zmqSendRequest({['command'] = 'new-batch', ['command-param'] = n})
end

function receiveBatch()
    -- wait for signal from Anglican that a new batch is ready
    io.write('Waiting for new batch...                                 \r')
    io.flush()
    local data = zmqReceiveReply()

    io.write('New batch received, processing batch...                  \r')
    io.flush()
    local b = getBatchTraces(data)

    io.write('New batch received, splitting into sub-batches...        \r')
    io.flush()
    local bs = getSubBatches(b)
    io.write('                                                         \r')
    io.flush()
    return bs
end

printLog('%{bluebg}%{bright white}Oxford Inference Compilation '..versionString)
printLog('%{bright white}Compilation mode')
printLog()
printLog('Started ' .. os.date('%a %d %b %Y %X'))
printLog()

args = ''
for i = 1, #arg do
    args = args .. ' ' .. arg[i]
end
printLog('Command line arguments:')
printLog(args)
printLog()

local prototype = nil
local parameters = nil
local gradParameters = nil
-- FUNCTION FOR RECONFIGURING, SAMPLE EMBEDDING LAYERS, PROPOSAL LAYERS, AND ARTIFACT PROTOTYPE
function reconfigurePrototype(batch)
    local hasChanged = false
    artifact.sampleLayers = artifact.sampleLayers or {}
    artifact.proposalLayers = artifact.proposalLayers or {}
    for _, subBatch in pairs(batch) do
        local exampleTrace = subBatch[1]
        local timeSteps = exampleTrace['timeSteps']
        local samples = exampleTrace['samples']
        for timeStep = 1, timeSteps do
            local address = samples[timeStep]['address']
            local instance = samples[timeStep]['instance']
            local value = samples[timeStep]['value']
            local smpDim = value:nElement()

            artifact.sampleLayers[address] = artifact.sampleLayers[address] or {}
            artifact.proposalLayers[address] = artifact.proposalLayers[address] or {}
            if not artifact.proposalLayers[address][instance] then
                artifact.sampleLayers[address][instance] = moveToCuda(nn.Linear(smpDim, artifact.smpEmbDim))

                local proposalType = samples[timeStep]['proposalType']
                local layer = nn.Sequential()
                if proposalType == 'categorical' then
                    -- unimplemented
                elseif proposalType == 'continuousminmax' then
                    layer:add(nn.Linear(artifact.lstmDim, 2))
                    local p = nn.Parallel(2, 1)
                    p:add(nn.Sigmoid())
                    p:add(nn.SoftPlus())
                    layer:add(p)
                    layer:add(nn.View(2, -1))
                    layer:add(nn.Transpose({1, 2}))
                elseif proposalType == 'dirichlet' then
                    local dim = samples[timeStep]['dim']
                    layer:add(nn.Linear(artifact.lstmDim, artifact.dirichletDim))
                    layer:add(nn.SoftPlus())
                elseif proposalType == 'discreteminmax' then
                    layer:add(nn.Linear(artifact.lstmDim, artifact.softMaxDim))
                    layer:add(nn.MulConstant(artifact.softMaxBoost, false))
                    layer:add(nn.SoftMax())
                elseif proposalType == 'flip' then
                    layer:add(nn.Linear(artifact.lstmDim, 1))
                    layer:add(nn.Sigmoid())
                elseif proposalType == 'foldednormal' then
                    layer:add(nn.Linear(artifact.lstmDim, 2))
                    layer:add(nn.SoftPlus())
                elseif proposalType == 'foldednormaldiscrete' then
                    layer:add(nn.Linear(artifact.lstmDim, 2))
                    layer:add(nn.SoftPlus())

                    -- TODO: solve this hack (which avoids numerical instability of distributions.norm.cdf)
                    layer:add(nn.MulConstant(4, false))
                elseif proposalType == 'mvn' then
                    local dim = samples[timeStep]['dim']
                    local p = nn.ConcatTable()
                    p:add(nn.Linear(artifact.lstmDim, dim))
                    p:add(
                        nn.Sequential()
                            :add(nn.Linear(artifact.lstmDim, dim * dim))
                            :add(nn.Sigmoid())
                            :add(nn.View(-1 ,dim, dim))
                    )
                    layer:add(p)
                elseif proposalType == 'mvnmeanvar' then
                    local dim = samples[timeStep]['dim']
                    local p = nn.ConcatTable()
                    p:add(nn.Linear(artifact.lstmDim, dim))
                    p:add(nn.Constant(0.001))
                    layer:add(p)
                -- elseif proposalType == 'mvnmeanvar' then
                --     local dim = samples[timeStep]['dim']
                --     local p = nn.ConcatTable()
                --     p:add(nn.Linear(artifact.lstmDim, dim))
                --     p:add(
                --         nn.Sequential()
                --             :add(nn.Linear(artifact.lstmDim, 1))
                --             :add(nn.SoftPlus())
                --     )
                --     layer:add(p)
                elseif proposalType == 'mvnmeanvars' then
                    local dim = samples[timeStep]['dim']
                    local p = nn.ConcatTable()
                    p:add(nn.Linear(artifact.lstmDim, dim))
                    p:add(
                        nn.Sequential()
                            :add(nn.Linear(artifact.lstmDim, dim))
                            :add(nn.SoftPlus())
                    )
                    layer:add(p)
                elseif proposalType == 'normal' or proposalType == 'laplace' then
                    layer:add(nn.Linear(artifact.lstmDim, 2))
                    local p = nn.Parallel(2, 1)
                    p:add(nn.Identity())
                    p:add(nn.SoftPlus())
                    layer:add(p)
                    layer:add(nn.View(2, -1))
                    layer:add(nn.Transpose({1, 2}))
                end

                artifact.proposalLayers[address][instance] = moveToCuda(layer)
                printLog('%{bright magenta}Polymorphing artifact, new layers attached : ' .. address .. ' ' .. instance)
                hasChanged = true
            end
        end
    end
    if (not prototype) or hasChanged then
        -- CREATE PROTOTYPE NN MODEL IN ORDER TO GET parameters, gradParameters, artifact.trainableParams
        prototype = nn.Sequential()

        -- Observe embedding layer
        prototype:add(artifact.observeLayer)

        -- Sample embedding layer
        for address, v in pairs(artifact.sampleLayers) do
            for instance, layer in pairs(v) do
                prototype:add(layer)
            end
        end

        -- LSTM layers
        for i = 1, artifact.lstmDepth do
            prototype:add(artifact.lstms[i])
        end

        -- proposal layer
        local p = nn.ParallelTable()
        for address, v in pairs(artifact.proposalLayers) do
            for instance, layer in pairs(v) do
                p:add(layer)
            end
        end
        prototype:add(p)
        parameters, gradParameters = prototype:getParameters()
        artifact.trainableParams = parameters:nElement()

        if opt.method == 'adam' then
            artifact.optimState = {
                learningRate = opt.learningRate,
                weightDecay = opt.weightDecay
            }
        elseif opt.method == 'sgd' then
            artifact.optimState = {
              learningRate = opt.learningRate,
              weightDecay = opt.weightDecay,
              momentum = opt.momentum,
              learningRateDecay = opt.learningRateDecay,
            }
        end
        if hasChanged then
            printLog('%{bright magenta}Polymorphing artifact, new trainable params: ' .. formatThousand(artifact.trainableParams))
        end
    end
end


-- INITIALIZE A NEW ARTIFACT OR RESUME AN EXISTING ARTIFACT
local prevArtifactTotalTrainingTime = 0
local prevArtifactTotalIterations = 0
local prevArtifactTotalTraces = 0
if opt.resume == '' then -- No artifact to resume, initialize a new artifact
    printLog('%{bright blue}** New artifact')
    printLog(string.format('File name             : %s', artifactFile))

    artifact.observeLayer = nil
    artifact.sampleLayers = {}
    artifact.lstms = {}
    artifact.proposalLayers = {}
    artifact.oneHotDict = nil
    artifact.trainLossBest = 1/0
    artifact.trainLossWorst = -1/0
    artifact.validLossBest = nil
    artifact.validLossWorst = nil
    artifact.validLossLast = nil
    artifact.validLossInitial = nil
    artifact.validLossFinal = nil
    artifact.validHistoryTrace = {}
    artifact.validHistoryLoss = {}
    artifact.trainHistoryTrace = {}
    artifact.trainHistoryLoss = {}
    artifact.trainableParams = nil
    artifact.obsEmb = opt.obsEmb
    artifact.obsEmbDim = opt.obsEmbDim
    artifact.smpEmb = opt.smpEmb
    artifact.smpEmbDim = opt.smpEmbDim
    artifact.lstmInputDim = 0
    artifact.lstmDim = opt.lstmDim
    artifact.lstmDepth = opt.lstmDepth
    artifact.softMaxBoost = opt.softMaxBoost
    artifact.softMaxDim = opt.softMaxDim
    artifact.dirichletDim = opt.dirichletDim
    artifact.oneHotDimAddress = opt.oneHotDim
    artifact.oneHotDimInstance = opt.oneHotDim
    artifact.noStandardize = opt.noStandardize
    artifact.obsSmooth = opt.obsSmooth
    artifact.oneHotDimProposalType = 10 -- This value we do know because it's the number proposal types that we support. It should be updated each time we add support for a new proposal type.
    artifact.dropout = opt.dropout
    artifact.created = os.date('%a %d %b %Y %X')
    artifact.codeVersion = versionString
    artifact.cuda = opt.cuda
    artifact.totalTrainingTime = 0
    artifact.totalIterations = 0
    artifact.totalTraces = 0
    artifact.updates = 0

    -- OBTAIN VALIDATION BATCH
    artifact.validSize = opt.validSize
    requestBatch(artifact.validSize)
    artifact.validBatch = receiveBatch()

    -- Create observe embedding layer
    local observeShape = nil
    local observeLength = 0
    for _, subBatch in pairs(artifact.validBatch) do
        local o = subBatch[1]['observes']
        observeShape = o:size()
        observeLength = o:nElement()
        break
    end

    local oemb = nn.Sequential()
    if artifact.obsEmb == 'lenet' then
        local H = observeShape[1]
        local W = observeShape[2]
        local K = 5
        local M = 2
        oemb:add(nn.JoinTable(1, 2))
        oemb:add(nn.View(-1, 1, H, W)) -- bx1x50x200
        oemb:add(nn.SpatialConvolution(1, 20, K, K)) -- bx20x46x196
        oemb:add(nn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.3)) end
        oemb:add(nn.SpatialMaxPooling(M, M, M, M)) -- bx20x23x98
        oemb:add(nn.SpatialConvolution(20, 50, K, K)) -- bx50x19x94
        oemb:add(nn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(nn.SpatialMaxPooling(M, M, M, M)) -- bx50x9x47
        oemb:add(nn.View(-1):setNumInputDims(3)) -- bx(50 * 9 * 47) = bx21150
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(50 * torch.floor((torch.floor((H - (K - 1)) / M) - (K - 1)) / M) * torch.floor((torch.floor((W - (K - 1)) / M) - (K - 1)) / M), artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(nn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(nn.ReLU(true))
    elseif artifact.obsEmb == 'cnn6' then
        -- TO DO: should be updated to work with observeShape other than 50x200
        oemb:add(nn.JoinTable(1, 2))
        oemb:add(nn.View(-1, 1, 50, 200)) -- bx1x50x200
        oemb:add(cudnn.SpatialConvolution(1, 64, 3, 3)) -- bx64x48x198
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.3)) end
        oemb:add(cudnn.SpatialConvolution(64, 64, 3, 3)) -- bx64x46x196
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.3)) end
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx64x23x98

        oemb:add(cudnn.SpatialConvolution(64, 128, 3, 3)) -- bx128x21x96
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x19x94
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x17x92
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx128x8x46

        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x6x44
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx128x3x22
        oemb:add(nn.View(-1):setNumInputDims(3)) -- bx(128 * 3 * 22) = bx8448
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(8448, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))

    elseif artifact.obsEmb == 'cnn6-100x100' then
        oemb:add(nn.JoinTable(1, 2))
        oemb:add(nn.View(-1, 1, 100, 100)) -- bx1x100x100
        oemb:add(cudnn.SpatialConvolution(1, 64, 3, 3)) -- bx64x98x98
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialConvolution(64, 64, 3, 3)) -- bx64x96x96
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx64x48x48

        oemb:add(cudnn.SpatialConvolution(64, 128, 3, 3)) -- bx128x46x46
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x44x44
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x42x42
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx128x21x21

        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x19x19
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx128x9x9
        oemb:add(nn.View(-1):setNumInputDims(3)) -- bx(128 * 3 * 22) = bx10368
        oemb:add(nn.Linear(10368, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
    elseif artifact.obsEmb == 'cnn6-96x96' then
        oemb:add(nn.JoinTable(1, 2))
        oemb:add(nn.View(-1, 1, 96, 96)) -- bx1x96x96
        oemb:add(cudnn.SpatialConvolution(1, 64, 3, 3)) -- bx64x94x94
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialConvolution(64, 64, 3, 3)) -- bx64x92x92
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx64x46x46
        --
        oemb:add(cudnn.SpatialConvolution(64, 128, 3, 3)) -- bx128x44x44
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x42x42
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x40x40
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx128x20x20
        --
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3)) -- bx128x18x18
        oemb:add(cudnn.ReLU(true))
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2)) -- bx128x9x9
        oemb:add(nn.View(-1):setNumInputDims(3)) -- bx(128 * 3 * 22) = bx10368
        oemb:add(nn.Linear(10368, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
    elseif artifact.obsEmb == 'cnn7' then
        -- TO DO: should be updated to work with observeShape other than 50x200
        oemb:add(nn.JoinTable(1, 2))
        oemb:add(nn.View(-1, 1, 50, 200)) -- bx1x50x200
        oemb:add(cudnn.SpatialConvolution(1, 64, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.3)) end
        oemb:add(cudnn.SpatialConvolution(64, 64, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.3)) end
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2))

        oemb:add(cudnn.SpatialConvolution(64, 128, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialConvolution(128, 128, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialMaxPooling(2,2,2,2))

        oemb:add(cudnn.SpatialConvolution(128, 256, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialConvolution(256, 256, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialConvolution(256, 256, 3, 3))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        oemb:add(cudnn.SpatialMaxPooling(2,2,1,1)) -- bx256x2x40
        oemb:add(nn.View(-1):setNumInputDims(3)) -- bx(256 * 2 * 40) = bx20480
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(20480, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
    elseif artifact.obsEmb == 'vgg' then
        -- TO DO: should be updated to work with observeShape other than 50x200
        oemb:add(nn.JoinTable(1, 2))
        oemb:add(nn.View(-1, 1, 50, 200)) -- bx1x50x200
        local function ConvBNReLU(nInputPlane, nOutputPlane)
            oemb:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
            oemb:add(cudnn.SpatialBatchNormalization(nOutputPlane, 1e-3))
            oemb:add(cudnn.ReLU(true))
        end
        ConvBNReLU(1, 64)
        if artifact.dropout then oemb:add(nn.Dropout(0.3)) end
        ConvBNReLU(64, 64)
        oemb:add(cudnn.SpatialMaxPooling(2, 2, 2, 2):ceil())

        ConvBNReLU(64, 128)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(128, 128)
        oemb:add(cudnn.SpatialMaxPooling(2, 2, 2, 2):ceil())

        ConvBNReLU(128, 256)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(256, 256)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(256, 256)
        oemb:add(cudnn.SpatialMaxPooling(2, 2, 2, 2):ceil())

        ConvBNReLU(256, 512)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(512, 512)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(512, 512)
        oemb:add(cudnn.SpatialMaxPooling(2, 2, 2, 2):ceil())

        ConvBNReLU(512, 512)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(512, 512)
        if artifact.dropout then oemb:add(nn.Dropout(0.4)) end
        ConvBNReLU(512, 512)
        oemb:add(cudnn.SpatialMaxPooling(2, 2, 2, 2):ceil()) -- bx512x2x7

        oemb:add(nn.View(-1):setNumInputDims(3)) -- bx(512 * 2 * 7) = bx7168
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(7168, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.BatchNormalization(artifact.obsEmbDim))
        oemb:add(cudnn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim)) -- bx(artifact.obsEmbDim)
        oemb:add(cudnn.ReLU(true))
    elseif artifact.obsEmb == 'cnn2-1d' then
        oemb:add(nn.JoinTable(1))
        oemb:add(nn.View(-1, 1, 1, 100))
        oemb:add(nn.SpatialConvolution(1,16,3,1))
        oemb:add(nn.ReLU(true))
        oemb:add(nn.SpatialConvolution(16,32,3,1))
        oemb:add(nn.ReLU(true))
        oemb:add(nn.SpatialMaxPooling(2,1,2,1))
        oemb:add(nn.View(-1):setNumInputDims(3))
        oemb:add(nn.Linear(1536, artifact.obsEmbDim))
        oemb:add(nn.ReLU(true))
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim))
        oemb:add(nn.ReLU(true))
    elseif artifact.obsEmb == 'fc' then
        oemb:add(nn.JoinTable(1))
        oemb:add(nn.View(-1, observeLength))
        oemb:add(nn.Linear(observeLength, artifact.obsEmbDim))
        oemb:add(nn.ReLU(true))
        if artifact.dropout then oemb:add(nn.Dropout(0.5)) end
        oemb:add(nn.Linear(artifact.obsEmbDim, artifact.obsEmbDim))
        oemb:add(nn.ReLU(true))
    else
        printLog('%{bright red}Unknown observe embedder: ' .. artifact.obsEmb)
        error()
    end
    -- cudnn.convert(oemb, cudnn)
    artifact.observeLayer = moveToCuda(oemb)

    -- Create LSTMs
    -- "nn.FastLSTM.bn" for batch normalization caused invariant lstm output (don't use without further investigation):
    -- nn.FastLSTM.bn = true
    nn.FastLSTM.usenngraph = true -- This is faster

    local lstmInputDim = artifact.obsEmbDim + artifact.smpEmbDim + artifact.oneHotDimAddress + artifact.oneHotDimInstance + artifact.oneHotDimProposalType

    artifact.lstms[1] = moveToCuda(nn.FastLSTM(lstmInputDim, artifact.lstmDim))
    if artifact.lstmDepth > 1 then
        for i = 2, artifact.lstmDepth do
            artifact.lstms[i] = moveToCuda(nn.FastLSTM(artifact.lstmDim, artifact.lstmDim))
        end
    end

    reconfigurePrototype(artifact.validBatch)
    -- printLog(tostring(prototype))
    printLog()

else -- Load an existing artifact for resuming training
    if (opt.resume == '') or (not opt.resume) then
        printLog('%{bright red}Error: Artifact to resume not found.\n')
    end

    artifact, artifactInfo = loadArtifact(opt.resume)
    prevArtifactTotalTrainingTime = artifact.totalTrainingTime
    prevArtifactTotalIterations = artifact.totalIterations
    prevArtifactTotalTraces = artifact.totalTraces

    printLog('%{bright blue}** Existing artifact')
    printLog(artifactInfo)
    printLog()
    printLog('%{bright blue}** New artifact')
    printLog(string.format('File name             : %s', artifactFile))
    printLog()

    if opt.validNew then
        -- GET NEW VALIDATION BATCH
        artifact.validSize = opt.validSize
        requestBatch(artifact.validSize)
        artifact.validBatch = receiveBatch()
    end
end


-- CREATE LOSS CRITERION
function proposalLoss(proposalOutput, subBatch)
    local numTraces = #subBatch
    local exampleTrace = subBatch[1]
    local timeSteps = exampleTrace['timeSteps']
    local logpdf = 0
    for timeStep = 1, timeSteps do
        local proposalType = exampleTrace['samples'][timeStep].proposalType
        if proposalType == 'categorical' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value
                local categories = subBatch[trace]['samples'][timeStep].categories
                local oneBasedIndex = tableInvert(categories)[value]
                local numCategories = subBatch[trace]['samples'][timeStep].numCategories
                local unnormalisedWeights = proposalOutput[timeStep][trace][{{1, numCategories}}]
                local weights = unnormalisedWeights / torch.sum(unnormalisedWeights)
                local logWeights = torch.log(epsilon + weights)
                local logWeight = logWeights[oneBasedIndex]

                logpdf = logpdf + logWeight
            end
        elseif proposalType == 'continuousminmax' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value[1]
                local min = subBatch[trace]['samples'][timeStep].min
                local max = subBatch[trace]['samples'][timeStep].max
                local normalisedValue = (value - min) / (max - min)
                -- TODO: put outside of the for loop for performance
                local normalisedMode = proposalOutput[timeStep][trace][1]
                local normalisedCertainty = proposalOutput[timeStep][trace][2] + 2
                local alpha = normalisedMode * (normalisedCertainty - 2) + 1
                local beta = (1 - normalisedMode) * (normalisedCertainty - 2) + 1

                logpdf = logpdf - torch.log(epsilon + cephes.beta(alpha, beta))
                    + (alpha - 1) * torch.log(epsilon + normalisedValue)
                    + (beta - 1) * torch.log(epsilon + 1 - normalisedValue)
                    - torch.log(epsilon + (max - min))
            end
        elseif proposalType == 'dirichlet' then
            -- TODO: UNEXPECTED ERRORS SUCH AS SEGMENTATION FAULT 11
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value
                local dim = subBatch[trace]['samples'][timeStep].alphaDim
                local unboundedAlpha = proposalOutput[timeStep][trace]
                local alpha = unboundedAlpha[{{1, dim}}]

                logpdf = logpdf + dists.dirichletLogpdf(value, alpha)
            end
        elseif proposalType == 'discreteminmax' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value[1]
                local min = subBatch[trace]['samples'][timeStep].min
                local max = subBatch[trace]['samples'][timeStep].max
                -- TODO: reduce dimension using the Resample layer
                local unnormalisedWeights = proposalOutput[timeStep][trace][{{1, max - min}}]
                local weights = unnormalisedWeights / torch.sum(unnormalisedWeights)
                local logWeights = torch.log(epsilon + weights)

                logpdf = logpdf + logWeights[value - min + 1]
            end
        elseif proposalType == 'flip' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value
                local prob = proposalOutput[timeStep][trace][1]
                local logprob = 0
                if value then
                    logprob = torch.log(prob)
                else
                    logprob = torch.log(1 - prob)
                end

                logpdf = logpdf + logprob
            end
        elseif proposalType == 'foldednormal' then
            local means = proposalOutput[timeStep][{{}, 1}]
            local SDs = proposalOutput[timeStep][{{}, 2}]
            local twoSDsquares = 2 * torch.cmul(SDs, SDs)
            local twoPiSDsquares = math.pi * twoSDsquares
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value[1]
                local mean = means[trace]
                local twoSDsquare = twoSDsquares[trace]
                local twoPiSDsquare = twoPiSDsquares[trace]
                local normLogpdf = -0.5 * torch.log(twoPiSDsquare) - torch.pow(value - mean, 2) / twoSDsquare
                local normLogpdfMinusValue = -0.5 * torch.log(twoPiSDsquare) - torch.pow(-value - mean, 2) / twoSDsquare
                local l = 0
                if value < 0 then
                    l = torch.log(epsilon)
                else
                    l = torch.log(torch.exp(normLogpdf) + torch.exp(normLogpdfMinusValue))
                end
                logpdf = logpdf + l
            end
        elseif proposalType == 'foldednormaldiscrete' then
            local means = proposalOutput[timeStep][{{}, 1}]
            local SDs = proposalOutput[timeStep][{{}, 2}]
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value[1]
                local mean = means[trace]
                local sd = SDs[trace]

                -- TODO: numerical issues with the foldedNormalDiscreteLogpdf function
                logpdf = logpdf + dists.foldedNormalDiscreteLogpdf(value, mean, sd)
            end
        elseif proposalType == 'mvn' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value
                local dim = subBatch[trace]['samples'][timeStep].dim
                local mean = proposalOutput[timeStep][1][trace]
                local preCov = proposalOutput[timeStep][2][trace]
                local cov = preCov + torch.t(preCov) + dim * moveToCuda(torch.eye(dim))

                logpdf = logpdf + dists.mvnLogpdf(value, mean, cov)
            end
        -- elseif proposalType == 'mvnmeanvar' then
        --     for trace = 1, numTraces do
        --         local value = subBatch[trace]['samples'][timeStep].value
        --         local dim = subBatch[trace]['samples'][timeStep].dim
        --
        --         local mean = proposalOutput[timeStep][1][trace]
        --         local var = proposalOutput[timeStep][2][trace]
        --         local cov = var[1] * moveToCuda(torch.eye(dim))
        --
        --         logpdf = logpdf + dists.mvnLogpdf(value, mean, cov)
        --     end
        elseif proposalType == 'mvnmeanvar' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value
                local dim = subBatch[trace]['samples'][timeStep].dim
                local mean = proposalOutput[timeStep][1][trace]
                local var = proposalOutput[timeStep][2]
                local cov = var[1] * moveToCuda(torch.eye(dim))

                logpdf = logpdf + dists.mvnLogpdf(value, mean, cov)
            end
        elseif proposalType == 'mvnmeanvars' then
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value
                local dim = subBatch[trace]['samples'][timeStep].dim
                local mean = proposalOutput[timeStep][1][trace]
                local vars = proposalOutput[timeStep][2][trace]

                -- Work around since torch.diag is unimplmented in Torch Autograd
                -- Consider implementing and submitting a pull request
                local cov = moveToCuda(torch.zeros(dim, dim))
                for d = 1, dim do
                    -- This is the same problem of Torch Autograd that we encountered in the very first version of the code
                    -- Can't assign values
                    cov[d][d] = vars[d].value
                end

                logpdf = logpdf + dists.mvnLogpdf(value, mean, cov)
            end
        elseif (proposalType == 'normal') then
            local means = proposalOutput[timeStep][{{}, 1}]
            local SDs = proposalOutput[timeStep][{{}, 2}]
            local twoSDsquares = 2 * torch.cmul(SDs, SDs)
            local twoPiSDsquares = math.pi * twoSDsquares
            for trace = 1, numTraces do
                local value = subBatch[trace]['samples'][timeStep].value[1]
                local mean = means[trace]
                local twoSDsquare = twoSDsquares[trace]
                local twoPiSDsquare = twoPiSDsquares[trace]
                logpdf = logpdf - 0.5 * torch.log(epsilon + twoPiSDsquare) - torch.pow(value - mean, 2) / twoSDsquare
            end
        end
    end
    return -logpdf / numTraces
end
local proposalCriterion = autograd.nn.AutoCriterion('Proposal loss')(proposalLoss)
if opt.cuda then proposalCriterion:cuda() end


printLog('%{bright blue}** Compilation configuration')
printLog('server                : ' .. opt.server)
printLog('log                   : ' .. opt.log)
printLog('artifact              : ' .. artifactFile)
printLog('keepArtifacts         : ' .. tostring(opt.keepArtifacts))
printLog('resumeLatest          : ' .. tostring(opt.resumeLatest))
printLog('resume                : ' .. opt.resume)
printLog('cuda                  : ' .. tostring(opt.cuda))
printLog('device                : ' .. opt.device)
printLog('validInterval         : ' .. opt.validInterval)
printLog('validSize             : ' .. artifact.validSize)
printLog('batchSize             : ' .. opt.batchSize)
printLog('method                : ' .. opt.method)
printLog('learningRate          : ' .. opt.learningRate)
printLog('learningRateDecay     : ' .. opt.learningRateDecay)
printLog('momentum              : ' .. opt.momentum)
printLog('weightDecay           : ' .. opt.weightDecay)
printLog('gradClip              : ' .. opt.gradClip)
printLog('lstmDim               : ' .. artifact.lstmDim)
printLog('lstmDepth             : ' .. artifact.lstmDepth)
printLog('obsEmb                : ' .. artifact.obsEmb)
printLog('obsEmbDim             : ' .. artifact.obsEmbDim)
printLog('smpEmb                : ' .. artifact.smpEmb)
printLog('smpEmbDim             : ' .. artifact.smpEmbDim)
printLog('softMaxBoost          : ' .. artifact.softMaxBoost)
printLog('softMaxDim            : ' .. artifact.softMaxDim)
printLog('dirichletDim          : ' .. artifact.dirichletDim)
printLog('oneHotDimAddress      : ' .. artifact.oneHotDimAddress)
printLog('oneHotDimInstance     : ' .. artifact.oneHotDimInstance)
printLog('oneHotDimProposalType : ' .. artifact.oneHotDimProposalType)
printLog()

function clearState()
    artifact.observeLayer:clearState()
    for address, v in pairs(artifact.sampleLayers) do
        for instance, layer in pairs(v) do
            layer:clearState()
        end
    end
    for i = 1, artifact.lstmDepth do
        artifact.lstms[i]:clearState()
    end
    for address, v in pairs(artifact.proposalLayers) do
        for instance, layer in pairs(v) do
            layer:clearState()
        end
    end
end

-- CLOSURE TO BE USED WITH OPTIM PACKAGE: DOES FORWARD AND BACKWARD PROPAGATION
function batchEvaluator(subBatch, backProp)
    local feval = function(x)
        collectgarbage()

        if x ~= parameters then
            parameters:copy(x)
        end
        gradParameters:zero()

        -- Prepare input to neural net
        local exampleTrace = subBatch[1]
        local timeSteps = exampleTrace['timeSteps']

        -- obs is a 1d or 2d valued table indexed by the trace number
        local obs = {}
        for trace = 1, #subBatch do
            obs[trace] = subBatch[trace]['observes']
        end

        -- smp is a 2d (number-of-traces x previous-sample-value) valued table indexed by the timestep in the trace
        -- TO DO: the following can fail if samples in different traces in the subbatch have different dimensions. We are not hashing and subbatching the traces according to that. Check further.
        local smp = {}
        smp[1] = Tensor(#subBatch, 1)
        for trace = 1, #subBatch do
            smp[1][trace] = 0 -- there is no previous sample for timestep 1
            for timeStep = 2, timeSteps do
                local prevSmpValue = subBatch[trace]['samples'][timeStep - 1].value
                local prevSmpDim = prevSmpValue:nElement()
                smp[timeStep] = smp[timeStep] or Tensor(#subBatch, prevSmpDim)
                smp[timeStep][trace] = prevSmpValue
            end
        end


        local model = nn.Sequential()

        -- Layers for inputting observes and samples
        local pt = nn.ParallelTable()

        -- Observe embedding
        pt:add(artifact.observeLayer)
        -- Sample embedding
        local semb = nn.ParallelTable()
        for timeStep = 1, timeSteps do
            local address = exampleTrace['samples'][timeStep].address
            local instance = exampleTrace['samples'][timeStep].instance
            local proposalType = exampleTrace['samples'][timeStep].proposalType
            local onehots = artifact.oneHotDict['address'][address]
            onehots = torch.cat(onehots, artifact.oneHotDict['instance'][instance])
            onehots = torch.cat(onehots, artifact.oneHotDict['proposalType'][proposalType])
            onehots = torch.view(onehots, 1, -1)
            onehots = torch.expand(onehots, #subBatch, onehots:nElement())

            if timeStep == 1 then
                semb:add(nn.Concat(2)
                            :add(nn.Constant(Tensor(#subBatch, artifact.smpEmbDim):zero()))
                            :add(nn.Constant(onehots)))
            else
                local prevAddress = exampleTrace['samples'][timeStep - 1].address
                local prevInstance = exampleTrace['samples'][timeStep - 1].instance
                semb:add(nn.Concat(2)
                            :add(artifact.sampleLayers[prevAddress][prevInstance])
                            :add(nn.Constant(onehots)))
            end
        end
        pt:add(semb)

        model:add(pt)

        -- Concatenating observes and samples
        local ct = nn.ConcatTable()
        for timeStep = 1, timeSteps do
            local s = nn.Sequential()

            -- Put in the same table
            local pt = nn.ParallelTable()
            pt:add(nn.Identity())
            pt:add(nn.SelectTable(timeStep))
            s:add(pt)

            -- Join Table
            s:add(nn.JoinTable(1, 1))

            ct:add(s)
        end
        model:add(ct)


        -- LSTM layer
        for i = 1, artifact.lstmDepth do
            model:add(nn.Sequencer(artifact.lstms[i]))
        end

        -- Proposal layer
        local p = nn.ParallelTable()
        for timeStep = 1, timeSteps do
            local address = exampleTrace['samples'][timeStep]['address']
            local instance = exampleTrace['samples'][timeStep]['instance']
            p:add(artifact.proposalLayers[address][instance])
        end
        model:add(p)
        moveToCuda(model)

        -- lstm:forget(): this is already done by the nn.Repeater/nn.Sequencer for each new sequence
        if backProp then
            model:training()
        else
            model:evaluate()
        end
        local proposalOutput = model:forward({obs, smp})
        local loss = proposalCriterion:forward(proposalOutput, subBatch)

        if backProp then
            local gradOutput = proposalCriterion:backward(proposalOutput, subBatch)
            model:backward({obs, smp}, gradOutput)

            if opt.gradClip ~= -1 then
                -- faster alternative (clamping)
                -- gradParameters:clamp(-opt.gradClip, opt.gradClip)

                local gn = gradParameters:norm()
                if gn > opt.gradClip then
                    gradParameters:mul(opt.gradClip / gn)
                end
            end

            for i = 1, #artifact.lstms do
                artifact.lstms[i]:forget()
            end
        end
        if opt.cuda then
            cutorch.synchronize()
        end

        return loss, gradParameters
    end
    return feval
end


-- TRAINING
local iteration = 0
local trace = 0
local startTime = os.time()
local improvementTime = os.time()

local trainLossString = ''

if not artifact.validLossBest then
    artifact.validLossBest = 0
    for _, subBatch in pairs(artifact.validBatch) do
        artifact.validLossBest = artifact.validLossBest + ((#subBatch) * batchEvaluator(subBatch, false)(parameters))
    end
    artifact.validLossBest = artifact.validLossBest / artifact.validSize
end
if not artifact.validLossWorst then artifact.validLossWorst = artifact.validLossBest end
if not artifact.validLossInitial then artifact.validLossInitial = artifact.validLossBest end
if prevArtifactTotalTraces == 0 then
    artifact.validHistoryTrace[#artifact.validHistoryTrace + 1] = prevArtifactTotalTraces + iteration
    artifact.validHistoryLoss[#artifact.validHistoryLoss + 1] = artifact.validLossBest
end
local validLossBestString = string.format('%+e', artifact.validLossBest)
local validLossString = string.format('%+e  ', artifact.validHistoryLoss[#artifact.validHistoryLoss])
local lastValidationTrace = 0

if opt.resume == '' then
    printLog('%{bright blue}** Training from ' .. opt.server)
else
    printLog('%{bright blue}** Resuming training from ' .. opt.server)
end
local timeString = daysHoursMinsSecs(prevArtifactTotalTrainingTime + (os.time() - startTime))
local improvementTimeString = daysHoursMinsSecs(os.time() - improvementTime)
local traceString = string.format('%5s', formatThousand(prevArtifactTotalTraces + trace))
printLog(string.format('%' .. string.len(timeString) .. 's', 'Train. time') .. ' │ ' .. string.format('%' .. string.len(traceString) .. 's', 'Trace') .. ' │ Training loss   │ Last valid. loss│ Best val. loss|' .. string.format('%' .. string.len(improvementTimeString) .. 's', 'T.since best'))
printLog(string.rep('─', string.len(timeString)) .. '─┼─' .. string.rep('─', string.len(traceString)) .. '─┼─────────────────┼─────────────────┼───────────────┼─' .. string.rep('─', string.len(improvementTimeString)))
requestBatch(opt.batchSize)
while true do
    local batch = receiveBatch()
    requestBatch(opt.batchSize)
    reconfigurePrototype(batch)

    -- io.write('Writing plot...                                  \r')
    -- io.flush()
    -- gnuplot.plot({'Training loss', trainLosses}, {'Validation loss', validLosses})

    for _, subBatch in pairs(batch) do
        iteration = iteration + 1
        io.write('Training...                                              \r')
        io.flush()

        local loss = nil
        if opt.method == 'sgd' then
            _, loss = optim.sgd(batchEvaluator(subBatch, true), parameters, artifact.optimState)
        elseif opt.method == 'adam' then
            _, loss = optim.adam(batchEvaluator(subBatch, true), parameters, artifact.optimState)
        end
        local trainLoss = loss[1]

        trace = trace + #subBatch

        artifact.totalTrainingTime = prevArtifactTotalTrainingTime + (os.time() - startTime)
        artifact.totalIterations = prevArtifactTotalIterations + iteration
        artifact.totalTraces = prevArtifactTotalTraces + trace

        artifact.trainHistoryTrace[#artifact.trainHistoryTrace + 1] = artifact.totalTraces
        artifact.trainHistoryLoss[#artifact.trainHistoryLoss + 1] = trainLoss

        if trainLoss < artifact.trainLossBest then
            artifact.trainLossBest = trainLoss
            trainLossString = '%{bright green}' .. string.format('%+e ▼', trainLoss) .. '%{reset}'
        elseif trainLoss > artifact.trainLossWorst then
            artifact.trainLossWorst = trainLoss
            trainLossString = '%{bright red}' .. string.format('%+e ▲', trainLoss) .. '%{reset}'
        elseif trainLoss < artifact.validHistoryLoss[#artifact.validHistoryLoss] then
            trainLossString = '%{green}' .. string.format('%+e  ', trainLoss) .. '%{reset}'
        elseif trainLoss > artifact.validHistoryLoss[#artifact.validHistoryLoss] then
            trainLossString = '%{red}' .. string.format('%+e  ', trainLoss) .. '%{reset}'
        else
            trainLossString = string.format('%+e  ', trainLoss)
        end

        timeString = daysHoursMinsSecs(prevArtifactTotalTrainingTime + (os.time() - startTime))

        traceString = string.format('%5s', formatThousand(prevArtifactTotalTraces + trace))
        if trace - lastValidationTrace > opt.validInterval then
            printLog(string.rep('─', string.len(timeString)) .. '─┼─' .. string.rep('─', string.len(traceString)) .. '─┼─────────────────┼─────────────────┼───────────────┼─' .. string.rep('─', string.len(improvementTimeString)))
            io.write('Computing validation loss...                             \r')
            io.flush()

            local validLoss = 0
            for _, subBatch in pairs(artifact.validBatch) do
                validLoss = validLoss + ((#subBatch) * batchEvaluator(subBatch, false)(parameters))
            end
            validLoss = validLoss / artifact.validSize
            lastValidationTrace = trace

            artifact.validHistoryTrace[#artifact.validHistoryTrace + 1] = artifact.totalTraces
            artifact.validHistoryLoss[#artifact.validHistoryLoss + 1] = validLoss

            if validLoss < artifact.validLossBest then
                artifact.validLossBest = validLoss
                validLossBestString = '%{bright green}' .. string.format('%+e', artifact.validLossBest) .. '%{reset}'
                validLossString = '%{bright green}' .. string.format('%+e ▼', validLoss) .. '%{reset}'
                io.write('Updating best artifact on disk...                        \r')
                io.flush()
                artifact.validLossFinal = validLoss
                artifact.modified = os.date('%a %d %b %Y %X')
                artifact.updates = artifact.updates + 1

                if opt.keepArtifacts then artifactFile = opt.artifact .. getTimeStamp() end
                clearState()
                torch.save(artifactFile, artifact)
                improvementTime = os.time()
            elseif validLoss > artifact.validLossWorst then
                artifact.validLossWorst = validLoss
                validLossString = '%{bright red}' .. string.format('%+e ▲', validLoss) .. '%{reset}'
                validLossBestString = string.format('%+e', artifact.validLossBest)
            elseif validLoss < artifact.validHistoryLoss[#artifact.validHistoryLoss] then
                validLossString = '%{green}' .. string.format('%+e  ', validLoss) .. '%{reset}'
                validLossBestString = string.format('%+e', artifact.validLossBest)
            elseif validLoss > artifact.validHistoryLoss[#artifact.validHistoryLoss] then
                validLossString = '%{red}' .. string.format('%+e  ', validLoss) .. '%{reset}'
                validLossBestString = string.format('%+e', artifact.validLossBest)
            else
                validLossString = string.format('%+e  ', validLoss)
                validLossBestString = string.format('%+e', artifact.validLossBest)
            end
        end

        improvementTimeString = daysHoursMinsSecs(os.time() - improvementTime)
        printLog(string.format('%s │ %s │ %s │ %s │ %s | %s', timeString, traceString, trainLossString, validLossString, validLossBestString, improvementTimeString))
    end
end
