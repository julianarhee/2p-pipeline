% Run this to grab relevant experiment info after acquisition struct has
% been created.

function [D] = loadAnalysisInfo(session,experiment, varargin)
% -------------------------------------------------------------------------
% USER-INUPT:
% -------------------------------------------------------------------------
% session = '20161221_JR030W';
% experiment = 'retinotopy037Hz';
% analysis_no = 3;
% tefo = false;
switch length(varargin)
    case 1
        analysis_no = varargin{1};
        tefo = false;
    case 2
        analysis_no = varargin{1};
        tefo = varargin{2};
end
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

% 
% acquisitionName = D.acquisitionName;
% nChannels = D.nChannels;
% roiType = D.roiType;
% nTiffs = D.nTiffs;
% metaPath = D.metaPath;
% maskType = D.maskType;
D.maskPaths = D.maskInfo.maskPaths;
% slicesToUse = D.slices;
% channelIdx = D.channelIdx;

traceNames = dir(fullfile(D.tracesPath, '*.mat'));
D.traceNames = {traceNames(:).name}';

D.figDir = fullfile(D.datastructPath, 'figures');
if ~exist(D.figDir, 'dir')
    mkdir(D.figDir);
end

D.outputDir = fullfile(D.datastructPath, 'output');
if ~exist(D.outputDir, 'dir')
    mkdir(D.outputDir);
end
% 
% D.outputDir = outputDir;
% D.figDir = figDir;
% D.traceNames = traceNames;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');