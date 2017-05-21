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

D = load(fullfile(analysisDir, strcat(datastruct, '.mat')));
D.tefo = tefo;


% Set paths:
D.figDir = fullfile(D.datastructPath, 'figures');
if ~exist(D.figDir, 'dir')
    mkdir(D.figDir);
end

D.outputDir = fullfile(D.datastructPath, 'output');
if ~exist(D.outputDir, 'dir')
    mkdir(D.outputDir);
end
% 


% 
% acquisitionName = D.acquisitionName;
% nChannels = D.nChannels;
% roiType = D.roiType;
% nTiffs = D.nTiffs;
% metaPath = D.metaPath;
% maskType = D.maskType;
if isfield(D, 'maskInfo')
    if strfind(D.roiType, '3D')
        
        % DO this silly thing where create paths to individual mask files
        % for each slice/file -- this is just to be consistent with
        % manual/planar analysis and plotting. 
        % NOTE:  D.maskInfo.maskPaths will contains paths to 3D masks for
        % EACH file, while D.maskPaths will create paths for each
        % slice-file combo...
        
        tmppaths = dir(fullfile(D.datastructPath, 'masks', 'masks_*'));
        tmppaths = {tmppaths(:).name}';
        D.maskPaths = {};
        for m=1:length(tmppaths)
            D.maskPaths{end+1} = fullfile(D.datastructPath, 'masks', tmppaths{m});
        end
    else
        D.maskPaths = D.maskInfo.maskPaths;
    end
else
    D.maskPaths = D.maskInfo.maskPaths;
end

%  Save 3D masks to D stuct:
if strfind(D.roiType, '3D')
       fprintf('Creating hdf5 mask array for NDB...\n');
        % Just need to load 3D masks:
        % if length(D.maskInfo.maskPaths) > 1
        % TODO:  currently, nmf is run on EACH file (i.e., each file may have
        % different masks)... deal with this. [try:  use binary spatial comps
        % for A to enforce same locs. across files/runs]
       tmpmasks = load(D.maskInfo.maskPaths{4});
       if strcmp(D.roiType, '3Dcnmf')
           masks = struct()
           fprintf('Getting masks and ids for 3Dnmf\n')
           %masks.maskmat = full(tmpmasks.spatialcomponents);
           %masks.maskids = tmpmasks.roi3Didxs;
           maskmat = full(tmpmasks.spatialcomponents);
           maskids = tmpmasks.roi3Didxs;
           masks.maskmat = maskmat;
           masks.maskids = maskids;
       else
           masks.maskmat = full(double(tmpmasks.roiMat));
           masks.maskids = tmpmasks.roi3Didxs;
       end
       maskarrayPath = fullfile(D.outputDir, 'maskarary.h5');
       
       %hdf5save(maskarrayPath, 'masks', 'masks');
       h5create(maskarrayPath, '/maskmat', size(maskmat));
       h5write(maskarrayPath, '/maskmat', maskmat);
       h5create(maskarrayPath, '/maskids', size(maskids));
       h5write(maskarrayPath, '/maskids', maskids);

       maskarraymatPath = fullfile(D.outputDir, 'maskarray.mat')
       save(maskarraymatPath, '-struct', 'masks');       

       D.maskarrayPath = maskarrayPath;
       D.maskarraymatPath = maskarraymatPath;
       D.nRois = size(maskmat,2);
% elseif strcmp(D.roiType, 'manual3D') && ~isfield(D, 'maskarrayPath')
%        fprintf('Creating hdf5 mask array for NDB...\n');
% 
%        tmpmasks = load(D.maskInfo.maskPaths{1});
%        masks = full(tmpmasks.roiMat);
%        maskarrayPath = fullfile(D.outputDir, 'maskarary.h5');
% 
%        hdf5save(maskarrayPath, 'masks', 'masks');
%        D.maskarrayPath = maskarrayPath;
%        D.nRois = size(masks,2);
end
       


if isfield(D, 'tracesPath')
    traceNames = dir(fullfile(D.tracesPath, 'traces_Slice*.mat'));
    D.traceNames = {traceNames(:).name}';
end

if isfield(D, 'tracesPath')
    traceNames = dir(fullfile(D.tracesPath, 'traces3D_*.mat'));
    D.traceNames3D = {traceNames(:).name}';
end


% D.outputDir = outputDir;
% D.figDir = figDir;
% D.traceNames = traceNames;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
