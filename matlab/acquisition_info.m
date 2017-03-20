% Run this to grab relevant experiment info after acquisition struct has
% been created.


% -------------------------------------------------------------------------
% USER-INUPT:
% -------------------------------------------------------------------------
session = '20161221_JR030W';
experiment = 'retinotopy037Hz';%'rsvp';
analysis_no = 1;
tefo = false;

% -------------------------------------------------------------------------
% Auto-generate:
% -------------------------------------------------------------------------

source = '/nas/volume1/2photon/RESDATA';

if tefo
    source = [source '/TEFO'];
end
acquisitionDir = fullfile(source, session, experiment);

datastruct = sprintf('datastruct_%03d', analysis_no);
analysisDir = fullfile(acquisitionDir, 'analysis', datastruct);
% jyr:  change 'datastruct' dir to "analysis" so it aint confusing

D = load(fullfile(analysisDir, datastruct));
acquisitionName = D.acquisitionName;
nChannels = D.nChannels;
roiType = D.roiType;
nTiffs = D.nTiffs;
metaPath = D.metaPath;
maskType = D.maskType;
maskPaths = D.maskInfo.maskPaths;
slicesToUse = D.slices;
channelIdx = D.channelIdx;

tracesPath = D.tracesPath;
traceNames = dir(fullfile(tracesPath, '*.mat'));
traceNames = {traceNames(:).name}';

figDir = fullfile(analysisDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

outputDir = fullfile(analysisDir, 'output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

D.outputDir = outputDir;
D.figDir = figDir;
D.traceNames = traceNames;
save(fullfile(analysisDir, datastruct), '-append', '-struct', 'D');