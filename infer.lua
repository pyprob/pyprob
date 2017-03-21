--
-- COMPILED INFERENCE
-- Infer mode
--
-- Tuan-Anh Le, Atilim Gunes Baydin
-- tuananh@robots.ox.ac.uk; gunes@robots.ox.ac.uk
--
-- Department of Engineering Science
-- University of Oxford
--
-- May -- September 2016
--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Oxford Compiled Inference')
cmd:text('Inference mode')
cmd:text()
cmd:text('Options:')
cmd:option('--help', false, 'display this help')
cmd:option('--version', false, 'display version information')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--log', './artifacts/infer-log', 'file for logging')
cmd:option('--artifact', './artifacts/compile-artifact', 'file name of the artifact')
cmd:option('--latest', false, 'use the latest artifact file starting with the name given with --artifact')
cmd:option('--debug', false, 'print out debugging information as requests arrive')
cmd:option('--server', '*:6666', 'address and port to bind this inference server')
cmd:text()

opt = cmd:parse(arg or {})

if opt.help then
    cmd:help()
    do return end
end
opt.log = opt.log .. os.date('-%y%m%d-%H%M%S')
require 'util'
if opt.version then
    print('Oxford Compiled Inference')
    print('Inference mode')
    print(versionString)
    do return end
end

printLog('%{bluebg}%{bright white}Oxford Compiled Inference '..versionString)
printLog('%{bright white}Inference mode')
printLog()
printLog('Started ' .. os.date('%a %d %b %Y %X'))
printLog()

local zmqContext = zmq.init(1)
local zmqSocket = zmqContext:socket(zmq.REP)
zmqSocket:bind('tcp://'..opt.server)

function zmqReceiveRequest()
    local msgreq = zmqSocket:recv()
    local request = {}
    for _, v in mp.unpacker(msgreq) do
        request = v
    end
    return request
end

function zmqSendReply(data)
    local msgrep = mp.pack(data)
    zmqSocket:send(msgrep)
end

printLog('%{bright blue}** Parameters')
local paramsstring = ''
for k, v in pairs(opt) do
    printLog(tostring(k)..': ' ..tostring(v))
end
printLog()

if opt.latest then
    opt.artifact = latestFileStartingWith(opt.artifact)
end

if not io.open(opt.artifact, "r") then
    printLog('%{bright red}Cannot read artifact file: '.. opt.artifact)
    error()
end
local artifact, artifactInfo = loadArtifact(opt.artifact)
artifact.observeLayer:evaluate()
for address, v in pairs(artifact.sampleLayers) do
    for instance, layer in pairs(v) do
        layer:evaluate()
    end
end
for i = 1, artifact.lstmDepth do
    artifact.lstms[i]:evaluate()
end
for address, v in pairs(artifact.proposalLayers) do
    for instance, layer in pairs(v) do
        layer:evaluate()
    end
end

printLog('%{bright blue}** Artifact')
printLog(artifactInfo)
printLog()

printLog('%{bright blue}** Inference server running at ' .. opt.server)

local observe = nil
local observeEmbedding = nil
local timeStep = 0

while true do
    spin()
    local request = zmqReceiveRequest()

    local command = request['command']
    local commandParam = request['command-param']
    if command == 'observe-init' then
        timeStep = 0

        local s = torch.Storage(commandParam['data'])

        -- BIG HACK TO WORK AROUND THE FACT THAT conv2 is not implemented in cutorch:
        -- https://github.com/torch/cutorch/issues/70
        if artifact.obsSmooth then
            observe = torch.Tensor(torch.LongStorage(commandParam['shape']))
            observe:storage():copy(s)
            if not(artifact.noStandardize) then
                observe = standardize(observe)
            end

            require 'image'
            observe = moveToCuda(image.convolve(observe, artifact.obsSmoothKernel))
        else
            observe = Tensor(torch.LongStorage(commandParam['shape']))
            observe:storage():copy(s)
            if not(artifact.noStandardize) then
                observe = standardize(observe)
            end
        end

        observeEmbedding = artifact.observeLayer:forward({observe})

        if opt.debug then
            printLog('\nCommand: observe-init')
        end

        for i = 1, #(artifact.lstms) do
            artifact.lstms[i]:forget()
        end
        zmqSendReply('observe-received')
    elseif command == 'observe-embed' then
        local s = torch.Storage(commandParam['data'])
        observe = Tensor(torch.LongStorage(commandParam['shape']))
        observe:storage():copy(s)
        if not(artifact.noStandardize) then
            observe = standardize(observe)
        end
        if artifact.obsSmooth then
            require 'image'
            observe = image.convolve(observe, artifact.obsSmoothKernel)
        end

        emb = artifact.observeLayer:forward({observe})

        if opt.debug then
            printLog('\nCommand: observe-embed')
        end

        zmqSendReply(emb:totable())
    elseif command == 'proposal-params' then
        -- TO DO: The following can be improved for faster inference, we don't need to create a whole new sequential model for each proposal-params request
        timeStep = timeStep + 1

        local address = commandParam['sample-address']
        local instance = commandParam['sample-instance']
        local proposalType = commandParam['proposal-name']
        local prevAddress = commandParam['prev-sample-address']
        local prevInstance = commandParam['prev-sample-instance']
        local tempPrevValue = commandParam['prev-sample-value']
        local prevValue = nil
        if type(tempPrevValue) == 'table' then
            prevValue = Tensor(tempPrevValue)
        elseif type(tempPrevValue) == 'number' then
            prevValue = Tensor(1)
            prevValue[1] = tempPrevValue
        elseif type(tempPrevValue) == 'boolean' then
            if tempPrevValue then
                prevValue = Tensor({1})
            else
                prevValue = Tensor({0})
            end
        end

        -- Sample embedding layer
        local prevSampleEmbedding = nil
        if timeStep == 1 then
            prevSampleEmbedding = Tensor(artifact.smpEmbDim):zero()
        else
            if not artifact.sampleLayers[prevAddress][prevInstance] then
                -- TO DO: we can handle this more gracefully, without aborting.
                printLog('%{bright red}Artifact has no sample layer for: ' .. prevAddress .. ' ' .. prevInstance)
                error()
            end
            prevSampleEmbedding = artifact.sampleLayers[prevAddress][prevInstance]:forward(prevValue)
        end

        local onehots = artifact.oneHotDict['address'][address]
        onehots = torch.cat(onehots, artifact.oneHotDict['instance'][instance])
        onehots = torch.cat(onehots, artifact.oneHotDict['proposalType'][proposalType])

        -- Make LSTM input by concatenating the correct things
        local lstmInput = observeEmbedding[1]
        lstmInput = torch.cat(lstmInput, prevSampleEmbedding)
        lstmInput = torch.cat(lstmInput, onehots)
        lstmInput = torch.view(lstmInput, 1, -1)

        local model = nn.Sequential()

        -- LSTM layer
        for i = 1, #(artifact.lstms) do
            model:add(artifact.lstms[i])
        end

        -- Proposal layer
        if not artifact.proposalLayers[address][instance] then
            -- TO DO: we can handle this more gracefully, without aborting. We can return proposal parameters for another instance of the same address. Or we can reply with 'unknown address-instance', then Anglican would handle it by proposing something independent of the artifact.
            printLog('%{bright red}Artifact has no proposal layer for: ' .. address .. ' ' .. instance)
            error()
        end
        model:add(artifact.proposalLayers[address][instance])
        model:evaluate()
        local proposalOutput = model:forward(lstmInput)
        local proposalParamsMsg = nil
        if proposalType == 'mvn' then
            proposalParamsMsg = {proposalOutput[1][1]:totable(), proposalOutput[2][1]:totable()}
        -- elseif proposalType == 'mvnmeanvar' then
        --     proposalParamsMsg = {proposalOutput[1][1]:totable(), proposalOutput[2][1][1]}
        elseif proposalType == 'mvnmeanvar' then
            proposalParamsMsg = {proposalOutput[1][1]:totable(), proposalOutput[2][1]}
        elseif proposalType == 'mvnmeanvars' then
            proposalParamsMsg = {proposalOutput[1][1]:totable(), proposalOutput[2][1]:totable()}
        else
            proposalParamsMsg = proposalOutput[1]:totable()
        end
        if opt.debug then
            printLog(string.format('\nTime step: %i', timeStep))
            printLog(string.format('Command: proposal-params, %s, %s, %s, %s', address, instance, proposalType, prevValue))
            printLog('Returned proposal parameter:')
            printLog(proposalParamsMsg)
        end
        zmqSendReply(proposalParamsMsg)
    end
end

zmqSocket:close()
zmqContext:term()
