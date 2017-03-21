--
-- INFERENCE COMPILATION
-- Artifact info
--
-- Tuan-Anh Le, Atilim Gunes Baydin
-- tuananh@robots.ox.ac.uk; gunes@robots.ox.ac.uk
-- University of Oxford
-- May 2016 -- March 2017
--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Oxford Compiled Inference')
cmd:text('Artifact info')
cmd:text()
cmd:text('Options:')
cmd:option('--help', false, 'display this help')
cmd:option('--version', false, 'display version information')
cmd:option('--log', './artifacts/artifact-info-log', 'file for logging')
cmd:option('--artifact', './artifacts/compile-artifact', 'artifact file name')
cmd:option('--nth', -1, 'use the nth artifact file starting with the name given with --artifact')
cmd:option('--latest', false, 'use the latest artifact file starting with the name given with --artifact')
cmd:option('--plotLoss', '', 'file name to save the loss plot (pdf)')
cmd:option('--plotLossToScreen', false, 'show the loss plot on screen')
cmd:option('--saveTrainLoss', '', 'file name to save training losses (csv)')
cmd:option('--saveValidLoss', '', 'file name to save validation losses (csv)')
cmd:text()

opt = cmd:parse(arg or {})

if opt.help then
    cmd:help()
    do return end
end
opt.log = opt.log .. os.date('-%y%m%d-%H%M%S')
require 'util'
if opt.version then
    print('Oxford Inference Compilation')
    print('Artifact info')
    print(versionString)
    do return end
end
printLog('%{bluebg}%{bright white}Oxford Inference Compilation '..versionString)
printLog('%{bright white}Artifact info')
printLog()

if opt.nth ~= -1 then
    opt.artifact = fileStartingWith(opt.artifact, opt.nth)
elseif opt.latest then
    opt.artifact = fileStartingWith(opt.artifact, -1)
end

local _, info, trainHistoryTrace, trainHistoryLoss, validHistoryTrace, validHistoryLoss = loadArtifact(opt.artifact)

printLog(info)

if (opt.plotLoss ~= '') or opt.plotLossToScreen then
    tt = torch.Tensor(trainHistoryTrace)
    tl = torch.Tensor(trainHistoryLoss)
    vt = torch.Tensor(validHistoryTrace)
    vl = torch.Tensor(validHistoryLoss)

    if opt.plotLoss ~= '' then gnuplot.pdffigure(opt.plotLoss) end
    gnuplot.xlabel('Traces')
    gnuplot.ylabel('Loss')
    gnuplot.plot({'Training', tt, tl},{'Validation', vt, vl})
    gnuplot.grid(true)
    gnuplot.plotflush()
    if opt.plotLoss ~= '' then printLog('Loss plot saved to ' .. opt.plotLoss) end
end

if opt.saveTrainLoss ~= '' then
    local out = assert(io.open(opt.saveTrainLoss, 'w'))
    out:write('Trace, TrainingLoss\n')
    for i = 1, #trainHistoryTrace do
        out:write(trainHistoryTrace[i] .. ', ' .. trainHistoryLoss[i] .. '\n')
    end
    out:close()
    printLog('Training losses saved to ' .. opt.saveTrainLoss)
end

if opt.saveValidLoss ~= '' then
    local out = assert(io.open(opt.saveValidLoss, 'w'))
    out:write('Trace, ValidationLoss\n')
    for i = 1, #validHistoryTrace do
        out:write(validHistoryTrace[i] .. ', ' .. validHistoryLoss[i] .. '\n')
    end
    out:close()
    printLog('Validation losses saved to ' .. opt.saveValidLoss)
end
