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
analysisDir = fullfile(acquisitionDir, 'datastructs', datastruct);
% jyr:  change 'datastruct' dir to "analysis" so it aint confusing

D = load(fullfile(analysisDir, data_struct));
acquisitionName = D.acquisition_name;
nChannels = D.nchannels;
roiType = D.roi_type;
ntiffs = D.ntiffs;
metaPaths = D.metaPaths;
maskType = D.maskType;
slices_to_use = D.slices_to_use;

tracesDir = fullfile(analysisDir, 'traces');
tracePaths = dir(tracesDir);

figDir = fullfile(analysisDir, 'figures');
if ~exist(figDir, 'dir')
    mkdir(figDir);
end
