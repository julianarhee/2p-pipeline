% Run this to grab relevant experiment info after acquisition struct has
% been created.


% -------------------------------------------------------------------------
% USER-INUPT:
% -------------------------------------------------------------------------
session = '20161221_JR030W';
experiment = 'retinotopy037Hz';
analysis_no = 1;


% -------------------------------------------------------------------------
% Auto-generate:
% -------------------------------------------------------------------------

source = '/nas/volume1/2photon/RESDATA';

tefo = false;
if tefo
    source = [source '/TEFO'];
end
acquisitionDir = fullfile(source, session, experiment);

datastruct = sprintf('datastruct_%03d', analysis_no);
analysisDir = fullfile(acquisitionDir, 'analysis', datastruct);
% jyr:  change 'datastruct' dir to "analysis" so it aint confusing

D = load(fullfile(analysisDir, datastruct));
acquisitionName = D.acquisition_name;
nChannels = D.channels;
roiType = D.roiType;
ntiffs = D.ntiffs;
metaPaths = D.metaPaths;
maskType = D.maskType;
maskPaths = D.maskInfo.maskPaths;
slices_to_use = D.slices;

tracesDir = fullfile(analysisDir, 'traces');
tracePaths = dir(fullfile(tracesDir, '*.mat'));
tracePaths = {tracePaths(:).name}';

figDir = fullfile(analysisDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

outputDir = fullfile(analysisDir, 'output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end