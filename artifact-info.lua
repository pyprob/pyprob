--
-- COMPILED INFERENCE
-- Artifact info
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
cmd:text('Artifact info')
cmd:text()
cmd:text('Options:')
cmd:option('--help', false, 'display this help')
cmd:option('--version', false, 'display version information')
cmd:option('--log', '../../data/artifact-info-log', 'file for logging')
cmd:option('--artifact', '../../data/compile-artifact', 'artifact file name')
cmd:option('--latest', false, 'use the latest artifact file starting with the name given with --artifact')
cmd:option('--plot', '', 'file name to save the loss plot (saved in pdf format)')
cmd:option('--plotToScreen', false, 'show the loss plot on screen')
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
    print('Artifact info')
    print(versionString)
    do return end
end
printLog('%{bluebg}%{bright white}Oxford Compiled Inference '..versionString)
printLog('%{bright white}Artifact info')
printLog()

if opt.latest then
    opt.artifact = latestFileStartingWith(opt.artifact)
end

assert(io.open(opt.artifact, "r"), 'Cannot read artifact file '..opt.artifact)
local _, info, trainLossHistTrace, trainLossHistLoss, validLossHistTrace, validLossHistLoss = loadArtifact(opt.artifact)

printLog(info)

if (opt.plot ~= '') or opt.plotToScreen then
    tt = torch.Tensor(trainLossHistTrace)
    tl = torch.Tensor(trainLossHistLoss)
    vt = torch.Tensor(validLossHistTrace)
    vl = torch.Tensor(validLossHistLoss)

    if opt.plot ~= '' then gnuplot.pdffigure(opt.plot) end
    gnuplot.xlabel('Traces')
    gnuplot.ylabel('Loss')
    gnuplot.plot({'Training', tt, tl},{'Validation', vt, vl})
    gnuplot.grid(true)
    gnuplot.plotflush()
    if opt.plot ~= '' then printLog('Loss plot saved to ' .. opt.plot) end
end
