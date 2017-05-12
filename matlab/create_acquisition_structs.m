clear all;
clc;

%% DEFINE SOURCE DIRECTORY:

% Define source dir for current acquisition/experiment:

% 20161222_JR030W.

% sourceDir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/retinotopy1/';
%acquisitionName = 'fov1_bar037Hz_run4';
%extraTiffsExcluded = [];

% sourceDir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/gratings1';
% acquisitionName = 'fov1_gratings_10reps_run1';
% extraTiffsExcluded = [19];

% sourceDir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/gratings2';
% acquisitionName = 'fov1_gratings_20reps_run2';
% extraTiffsExcluded = [];


% 20161221_JR030W.

% ---- NMF: ---------------------------------------------------------------
% sourceDir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/test_crossref/nmf';
% acquisitionName = 'fov1_bar037Hz_run4';
% extraTiffsExcluded = [];
% tefo = false;

% datastruct_001 :  nmf, larg(er) neuron size specified, didn't save output
% datastruct_002 :  nmf, smaller neurons, better overlap size, saving
% output
% datastruct_003 :  nmf, same settings, run patch, but run on all files.
% datastruct_004 :  same source data as NMF, but do pixels to compare
% against TeFo, use sigma=3

% datastruct_005 :  bad-parsing..... re-do

% ---- 12k res ------------------------------------------------------------

%sourceDir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/retinotopy037Hz';
%acquisitionName = 'fov1_bar037Hz_run4';
%extraTiffsExcluded = [];

% sourceDir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/test_crossref';
% acquisitionName = 'test_crossref';
% extraTiffsExcluded = [];

% datastruct_001 : manual reois, retinotopy037Hz part
% datastruct_002 : same as _001, but for rsvp
% datastruct_003 : aborted attempt for manual 3D rois.


% ---- rsvp -----
%sourceDir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp';
%acquisitionName = 'fov2_rsvp_25reps';
%extraTiffsExcluded = [];

% datastruct_001 :  condition ROIs, from retinotopy037Hz...
% datastruct_002 :  manual ROIs, selected within itself

%sourceDir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp_run3';
%acquisitionName = 'fov2_rsvp_25reps_run3';
%extraTiffsExcluded = [9 10];


% ==============================
% DEFINE session info:
% =============================
%sessionpath = fullfile(source, session);
%sessionmeta = getSessionMeta(sessionpath, D.tefo);

% TEFO:  20161219_JR030W.

source = '/nas/volume1/2photon/RESDATA/TEFO';
session = '20161219_JR030W';
run = 'retinotopyFinal';

sourceDir = fullfile(source, session, run);

% -------------------------------------------------------------------------
% retinotopyFinal/
% -------------------------------------------------------------------------
%sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal';
acquisitionName = 'fov6_retinobar_037Hz_final_nomask';
extraTiffsExcluded = [];
tefo = true;

average = true;
matchtrials = [{[1, 1]}, {[2, 3]}, {[3, 2]}, {[4, 4]}];

% datastruct_001 :  manually-selected ROIs on every other slice, 13-25.
% datastruct_002 :  do pixel-wide analysis, sigma=3 (to compare to
% avg - datastruct_007)

% **datastruct_003 :  roiMap, AVG ROI MAP (compare with gratingsFinalMask2)
% - TODOOOO

% datastruct_004 :  3Dcnmf, + AVG ROI MAP, substack 9-30.

% datastruct_005 :  3Dcnmf + AVROIMAP + substack9-30. **average runs with retintopyFinalMask**
% --> This one is messed up, only 1 and 4 are basically run-twice (run),
% since averaging was only done with itself.  Bad file-indexing fixed.

% datastruct_006 :  3Dcnmf + AVGROIMAP + substacks9-30.  D.average
% **REDOING of datastruct_005 with correctly averaged trials.
% - max_size_thr slightly lowered to 8 (was 10)

% datastruct_007 :  same (3Dcnmf, AVGROIMAP, substacks9-30, D.average)
% - don't filter very small components... (ff) D.maskInfo.params.keepAll
% - set thr_method to 'max' and see if setting 'maxthr'=0.9 (default: 0.1)
% helps keep all...


% -------------------------------------------------------------------------
% retinotopyFinalMask/
% -------------------------------------------------------------------------
% sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinalMask';
% acquisitionName = 'fov6_retinobar_037Hz_final_bluemask';
% extraTiffsExcluded = [];
% tefo = true;

% datastruct_002 :  'condition;, used 'condition' ROIs from retinotopyFinal.
% datastruct_001 :  '3Dcnmf'
% datastruct_003 :  '3Dcnmf' - use new params for 3Dcnmf, use SUBSTACK.
% datastruct_004 :  same as _003, but different params
% datastruct_005 :  use x-ray on slice 18 of 27-slice stack...
% datastruct_006 :  same as _005, but more Rois selected, and only slices
% 6-30 (remove first 5 slices)
% datastruct_007 :  pixel analysis, remove first 5 slices, try with sigma=3
% (compare with retinotopyFinal - datastruct_002)

% datastruct_008 : Just get substack9-30 for memmap ??

% -------------------------------
% 
% sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyControl';
% acquisitionName = 'fov6_037Hz_nomask_bar1_shutteroff';
% extraTiffsExcluded = [];
% tefo = true;

% datastruct_001 :  pixels, sigma=3; compare retinotopyFinal and retinotopyFinalMask



% -- TEFO:  gratings/ rsvp -------------------------------------------------------
% 
% source = '/nas/volume1/2photon/RESDATA/TEFO';
% session = '20161219_JR030W';
% run = 'gratingsFinalMask2';
% 
% sourceDir = fullfile(source, session, run);
% 
% acquisitionName = 'fov6_gratings_bluemask_5trials';
% extraTiffsExcluded = [];
% tefo = true;

% too big?...
% sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratingsFinalMask2';
% acquisitionName = 'fov6_gratings_bluemask_5trials';
% extraTiffsExcluded = [];
% tefo = true;

% datastruct_001 : roiMap, use AVG ROI MAP (from retinotopy runs), then select
% using blob_detector (python); blobType = 'difference' 
% datastruct_002 : 3Dcnmf, use AVG ROI MAP (from averaged retinotopy runs) to seed
% 3DCNMF.


%
%
% sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvpFinal2';
% acquisitionName = 'fov6_rsvp_nomask_test_10trials';
% extraTiffsExcluded = [];


% sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/raw/bar5';


%%

analysisDir = fullfile(sourceDir, 'analysis');
if ~exist(analysisDir, 'dir')
    mkdir(analysisDir);
end

channelIdx = 1;     % Set channel with GCaMP activity (Channel01)

didx = 7;           % Define datastruct analysis no.

metaInfo = 'SI';    % Define source of meta info (usualy 'SI')
                    % options: 'manual' or 'SI'
                    % If 'manual' each entry needs to be filled in...
                    % This toggles whether tiff-source will be in
                    % "Corrected" or "Parsed" folder in Step 2 (when TIFF
                    % directorie

corrected = false;   % Flag for whether motion-corrected using standard Acquisition2P_Class or not

nchannels = 2;      % N channels acquired in session.

% --------------------------------------------
% 1. Specifiy preprocessing method:
% --------------------------------------------
%D.preprocessing = 'Acquisition2P';
preprocessing = 'raw';
%preprocessing = 'fiji';


% --------------------------------------------
% 2.a.  Specify ROI type for current analysis:
% --------------------------------------------
%roiType = 'create_rois';
% roiType = 'roiMap';
%roiType = 'manual3Drois';
%roiType = 'condition';
%roiType = 'pixels';
%D.roiType = 'cnmf';
roiType = '3Dcnmf'

% 2.b.  Specify additional args, depending on ROI type:
if strcmp(roiType, 'condition')
% Test cross-ref:
    refMaskStruct = 'retinotopyFinal'; %refMaskStruct = 'gratings1';
    refMaskStructIdx = 1; %refMaskStructIdx = 2;
elseif strcmp(roiType, 'roiMap')
    pathparts = strsplit(sourceDir, '/');
    mapSource = strjoin(pathparts(1:end-1), '/');
    mapSource = fullfile(mapSource, 'average_volumes');
    mapSlicesPath = fullfile(mapSource, 'avg_frames_conditions_channel01');
    roiPaths = dir(fullfile(mapSource, 'rois*.mat'));
    if length(roiPaths)==1
        roiPath = roiPaths(1).name;
    else
        roiIdx = 1; % Specify which ROI maps to use, if more than 1
        roiPath = roiPaths(roiIdx).name;
    end
elseif strcmp(roiType, '3Dcnmf')
    seedRois = true;
end

% --------------------------------------------
% 3.  Specify slices to process:
% --------------------------------------------
%slicesToUse = 7:17; 
%slicesToUse = [10, 15];  
%slicesToUse = [13:2:25];
% slicesToUse = [1:20];
%slicesToUse = [6:30];
%slicesToUse = [4:30];
%slicesToUse = [6:20];
slicesToUse = [9:30];


% 4.  Provide additional params:
metaonly = false;
nhugetiffs = 0;

%% Define datastruct name for analysis:

% didx = 1;

datastruct = sprintf('datastruct_%03d', didx);
dstructPath = fullfile(analysisDir, datastruct);
existingAnalyses = dir(analysisDir);
existingAnalyses = {existingAnalyses(:).name}';

if ~exist(dstructPath)
    mkdir(dstructPath)
    D = struct();
else
    fprintf('*********************************************************\n');
    fprintf('WARNING:  Specified datastruct -- %s -- exists. Overwrite?\n');
    uinput = input('Press Y/n to overwrite or create new: \n', 's');
    if strcmp(uinput, 'Y')
        D = struct();
        fprintf('New datastruct created: %s.\n', datastruct);
        fprintf('Not yet saved. Exit now to load existing datastruct.\n');
    else
        didx = input('Enter # to create new datastruct: \n');
        datastruct = sprintf('datastruct_%03d', didx);
        while ismember(datastruct, existingAnalyses)
            didx = input('Analysis %s already exists... Choose again.\n', datastruct);
            datastruct = sprintf('datastruct_%03d', didx);
        end
        dstructPath = fullfile(analysisDir, datastruct);
        mkdir(dstructPath);
        D = struct();
    end
end

D.name = datastruct;
D.datastructPath = dstructPath;
D.sourceDir = sourceDir;
D.acquisitionName = acquisitionName;
D.roiType = roiType;
D.preprocessing = preprocessing;
D.channelIdx = channelIdx;
D.extraTiffsExcluded = extraTiffsExcluded;
D.slices = slicesToUse;
D.tefo = tefo;

D.average = average;
if D.average
    D.matchtrials = matchtrials;
end

save(fullfile(dstructPath, datastruct), '-struct', 'D');

fprintf('Created new datastruct analysis: %s\n', D.datastructPath)
%% Get SI volume info:

% metaInfo = 'SI';
% 
% corrected = true;
if corrected
    D.tiffSource = 'Corrected';
else
    D.tiffSource = 'Parsed';
    D.metaonly = metaonly;
    if D.metaonly
        D.nTiffs = nhugetiffs;
    end
end

% Check if more than one "experiment" in current motion-correction
% acquisition (i.e., multiple conditions in session corrected to a single
% reference run of some other condition):
tmpDirs = dir(fullfile(D.sourceDir, D.tiffSource));
tmpDirs = tmpDirs(arrayfun(@(x) x.name(1), tmpDirs) ~= '.');
tmpDirs = tmpDirs(find(~cell2mat(strfind(arrayfun(@(x) x.name, tmpDirs, 'UniformOutput', 0),'Channel'))));

if numel(tmpDirs) > 1
    D.nExperiments = length(tmpDirs);
    D.experimentNames = arrayfun(@(x) tmpDirs(x).name, 1:length(tmpDirs), 'UniformOutput', 0);
    fprintf('More than 1 condition corrected with current refernce:\n');
    for eidx=1:length(D.experimentNames)
        fprintf('Idx: %i, Name: %s\n', eidx, D.experimentNames{eidx});
    end
    selectedExperiment = input('Select idx of condition to analyze:\n');
    D.tiffSource = fullfile(D.tiffSource, D.experimentNames{selectedExperiment});
    D.experiment = D.experimentNames{selectedExperiment};
else
    D.nExperiments = 1;
    D.experimentNames = acquisitionName;
    D.experiment = acquisitionName;
end
    
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

fprintf('Got SI file info.\n');

%% Get SI volume info:

switch metaInfo
    case 'SI'
       
        siMetaName = sprintf('%s.mat', D.acquisitionName);
        
        if ~exist(fullfile(D.sourceDir, siMetaName))
            
            fprintf('No motion-corrected META data found...\n');
            fprintf('Current acquisition is: %s\n', D.acquisitionName);
            fprintf('Parsing SI tiffs, and creating new meta file.\n');
            
            % Load and parse raw TIFFs and create SI-specific meta file:
            movies = dir(fullfile(D.sourceDir,'*.tif'));
            movies = {movies(:).name};
            writeDir = fullfile(D.sourceDir, D.tiffSource);
            if ~exist(writeDir, 'dir')
                mkdir(writeDir)
            end
            switch D.preprocessing
                case 'fiji'
                    fprintf('No SI metadata for fiji tiffs.\nSelect reference:\n');
                    [fn, fpath, ~] = uigetfile();
                    sitmp = load(fullfile(fpath, fn));
                    siRefAcquisitionName = fieldnames(sitmp);
                    siRef = sitmp.(siRefAcquisitionName{1});
                    fprintf('Selected reference si metastruct from: %s.\n', siRef.acqName);
                    if length(movies)==1
                        siRefFidx = input('Select FILE idx from current reference acquisition:\n');
                        siRefMetaStruct = siRef.metaDataSI{siRefFidx};
                        parseSIdata(D.acquisitionName, movies, D.sourceDir, writeDir, siRefMetaStruct);
                    else
                        siRefMetaStruct = siRef;
                        if length(movies) ~= length(siRef.metaDataSI)
                            siRefStartIdx = input('Select idx for FIRST file in reference meta stuct:\n');
                            if isempty(siRefStartIdx)
                                siRefStartIdx = 1;
                            end
                        else
                            siRefStartIdx = 1;
                        end
                        parseSIdata(D.acquisitionName, movies, D.sourceDir, writeDir, siRefMetaStruct, siRefStartIdx);
                    end
                    
                otherwise
                    if D.metaonly
                        parseSIdata(D.acquisitionName, movies, D.sourceDir, writeDir, [], [], D.metaonly);
                    else
                        parseSIdata(D.acquisitionName, movies, D.sourceDir, writeDir);
                    end
            end
            
            % Load newly-created meta struct:
            %siMeta = load(fullfile(sourceDir, siMetaName));
            %meta = struct();
            %meta.(acquisitionName) = siMeta.metaDataSI; % Have info for each file, but this was not corrected in previosuly run MCs...
        end
        
        % Sort Parsed files into separate directories if needed:
        tmpchannels = dir(fullfile(D.sourceDir, D.tiffSource));
        tmpchannels = tmpchannels(arrayfun(@(x) ~strcmp(x.name(1),'.'), tmpchannels));
        tmpchannels = tmpchannels([tmpchannels.isdir]);
        tmpchannels = {tmpchannels(:).name}';
        %if length(dir(fullfile(D.sourceDir, D.tiffSource, tmpchannels{1}))) > length(tmpchannels)+2
        if isempty(tmpchannels) || strcmp(D.tiffSource, 'Parsed')
            sort_parsed_tiffs(D, nchannels);
        end
        
        % Creata META with SI (acquisition) and MW (experiment) info:
        meta = createMetaStruct(D);
        fprintf('Created meta struct for current acquisition,\n'); 
        
        
    case 'manual' % No motion-correction/processing, just using raw TIFFs.
        nVolumes = 350;
        nSlices = 20;
        nDiscard = 0;
        
        nFramesPerVolume = nSlices + nDiscard;
        nTotalFrames = nFramesPerVolume * nVolumes;
        
        %.....

end


D.metaPath = meta.metaPath;
D.nTiffs = meta.nTiffs;
D.nChannels = meta.nChannels;
D.stimType = meta.stimType;

save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

 
%% Set up experiment analysis parameters:

% =========================================================================
% Parameters:
% =========================================================================

% 
% % 1. Specifiy preprocessing method:
% % --------------------------------------------
% D.preprocessing = 'Acquisition2P';
% % preprocessing = 'raw';
% 
% 
% % 2.  Specify ROI type for current analysis:
% % --------------------------------------------
% %D.roiType = 'create_rois';
% %D.roiType = 'condition';
% %D.roiType = 'pixels';
% D.roiType = 'cnmf';

switch D.roiType
    case 'create_rois'
        [fpath,fcond,~] = fileparts(D.sourceDir);
        D.maskSource = fcond;
    
    case 'roiMap'
        D.maskSource = mapSource;
        
    case 'manual3Drois'
        [fpath,fcond,~] = fileparts(D.sourceDir);
        D.maskSource = fcond;
        
    case 'condition'
        D.maskSource = refMaskStruct; %'retinotopy1';
        D.maskDidx = refMaskStructIdx;
        D.maskDatastruct = sprintf('datastruct_%03d', D.maskDidx);
        fprintf('Using pre-defined masks from condition: %s.\n', D.maskSource);
        
        [fpath,fcond,~] = fileparts(D.sourceDir);
        D.maskSourcePath = fullfile(fpath, D.maskSource);
        pathToMasks = fullfile(D.maskSourcePath, 'analysis', D.maskDatastruct, 'masks');
        maskNames = dir(fullfile(pathToMasks, 'masks_*'));
        maskNames = {maskNames(:).name}';
        D.maskPaths = cell(1, length(maskNames));
        for maskIdx=1:length(maskNames)
            D.maskPaths{maskIdx} = fullfile(pathToMasks, maskNames{maskIdx});
        end
        
    case 'pixels'
        [fpath,fcond,~] = fileparts(D.sourceDir);
        D.maskSource = fcond;
        
    case 'cnmf'
        [fpath,fcond,~] = fileparts(D.sourceDir);
        D.maskSource = fcond;
        
    case '3Dcnmf'
        if seedRois
            pathparts = strsplit(sourceDir, '/');
            mapSource = strjoin(pathparts(1:end-1), '/');
            mapSource = fullfile(mapSource, 'average_volumes');
            mapSlicesPath = fullfile(mapSource, 'avg_frames_conditions_channel01');
            roiPaths = dir(fullfile(mapSource, 'centroids*.mat'));
            if length(roiPaths)==1
                roiPath = roiPaths(1).name;
            else
                roiIdx = 1; % Specify which ROI maps to use, if more than 1
                roiPath = roiPaths(roiIdx).name;
            end
            D.maskSource = mapSource;
        else
            [fpath,fcond,~] = fileparts(D.sourceDir);
            D.maskSource = fcond;
        end
        
end

% =========================================================================
% Save analysis info:
% =========================================================================

save(fullfile(D.datastructPath, D.name), '-struct', 'D');

fprintf('Got source info for creating ROI masks.\n')

%%

% rois = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/average_volumes/rois_avg_frames_conditions_channel01.mat')
% 
% fig = figure(); imagesc(sliceim); colormap(gray);
% 
% currslice = 13;
% currslice_name = sprintf('slice%i', currslice);
% 
% currrois = rois.(currslice_name).DoG;
% radii = currrois(:,3);
% centers = [currrois(:,2), currrois(:,1)];
% 
% 
% fig = figure(); imagesc(sliceim); colormap(gray);
% hold on;
% viscircles(gca, centers, radii, 'LineWidth', 0.2, 'Color', 'y')


%% Create masks and get traces:

% =========================================================================
% Create masks, and save them:
% =========================================================================
tic()

% slicesToUse = [5, 10, 15, 20];                            % Specify which slices to use (if empty, will grab traces for all slices)

switch D.roiType
    case 'create_rois'
        
        % Choose reference:
        % -----------------------------------------------------------------
        D.slices = slicesToUse;                            % Specify which slices to use (if empty, will grab traces for all slices)

        refMeta = load(meta.metaPath);                          % Get all TIFFs (slices) associated with file and volume of refRun movie.
        if isfield(refMeta.file(1).si, 'motionRefNum')
            D.refRun = refMeta.file(1).si.motionRefNum;
            D.refPath = refMeta.file(D.refRun).si.tiffPath;
        else
            D.refRun = round(length(refMeta.file)/2);
            D.refPath = 'na';
        end
        
        % Create ROIs:
        % -----------------------------------------------------------------
        create_rois(D, refMeta);
                
        
        % Set up mask info struct to reuse masks across files:
        % -----------------------------------------------------------------
        D.maskType = 'circles';
        D.maskInfo = struct();
        D.maskInfo.refNum = D.refRun;
        D.maskInfo.refMeta = refMeta;
        D.maskInfo.maskType = D.maskType;

        maskDir = fullfile(D.datastructPath, 'masks');
        maskStructs = dir(fullfile(maskDir, '*.mat'));
        maskStructs = {maskStructs(:).name}';
        slicesToUse = zeros(1,length(maskStructs));
        for m=1:length(maskStructs)
            mparts = strsplit(maskStructs{m}, 'Slice');
            mparts = strsplit(mparts{2}, '_');
            slicesToUse(m) = str2num(mparts{1});
        end
        maskPaths = cell(1,length(maskStructs));
        for m=1:length(maskStructs)
            maskPaths{m} = fullfile(maskDir, maskStructs{m});
        end
        D.maskInfo.maskPaths = maskPaths;
        D.maskInfo.slices = slicesToUse;
        
        
        % =================================================================
        % Get traces with masks:
        % =================================================================
        tic()
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        toc()
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
    
    case 'roiMap'
        
        D.maskType = 'circles';
        D.maskInfo = struct();
        D.maskInfo.mapSource = mapSource;
        D.maskInfo.mapSlicePaths = mapSlicesPath;
        D.maskInfo.roiPath = roiPath; 
        D.maskInfo.slices = slicesToUse;
        D.maskInfo.blobType = 'difference';
        %D.maskInfo.refNum = 1;
        
        % Create masks from ROI maps: 
        % -----------------------------------------------------------------
        rois_to_masks(D);
        
        maskDir = fullfile(D.datastructPath, 'masks');
        maskStructs = dir(fullfile(maskDir, '*.mat'));
        maskStructs = {maskStructs(:).name}';
        for m=1:length(maskStructs)
            maskPaths{m} = fullfile(maskDir, maskStructs{m});
        end
        D.maskInfo.maskPaths = maskPaths;
        
        % =================================================================
        % Get traces with masks:
        % =================================================================
        tic()
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        toc()
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
    
        fprintf('Got traces for roiMap analysis (%s), %s', D.name, D.acquisitionName);
        
    case 'manual3Drois'
        
        % Choose reference:
        % -----------------------------------------------------------------
        D.slices = slicesToUse;                            % Specify which slices to use (if empty, will grab traces for all slices)

        refMeta = load(meta.metaPath);                          % Get all TIFFs (slices) associated with file and volume of refRun movie.
        D.refRun = refMeta.file(1).si.motionRefNum;
        D.refPath = refMeta.file(D.refRun).si.tiffPath;
        D.localRefNum = 4
        % Create ROIs:
        % -----------------------------------------------------------------
        create_rois(D, refMeta);
                
        
        % Set up mask info struct to reuse masks across files:
        % -----------------------------------------------------------------
        D.maskType = 'circles';
        D.maskInfo = struct();
        D.maskInfo.refNum = D.refRun;
        D.maskInfo.refMeta = refMeta;
        D.maskInfo.maskType = D.maskType;

        maskDir = fullfile(D.datastructPath, 'masks');
        maskStructs = dir(fullfile(maskDir, '*.mat'));
        maskStructs = {maskStructs(:).name}';
        slicesToUse = zeros(1,length(maskStructs));
        for m=1:length(maskStructs)
            mparts = strsplit(maskStructs{m}, 'Slice');
            mparts = strsplit(mparts{2}, '_');
            slicesToUse(m) = str2num(mparts{1});
        end
        maskPaths = cell(1,length(maskStructs));
        for m=1:length(maskStructs)
            maskPaths{m} = fullfile(maskDir, maskStructs{m});
        end
        D.maskInfo.maskPaths = maskPaths;
        D.maskInfo.slices = slicesToUse;
        
        
        % =================================================================
        % Get traces with masks:
        % =================================================================
        tic()
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        toc()
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
    case 'condition'
        % Use masks from a different condition:
        ref = load(fullfile(D.maskSourcePath, 'analysis', sprintf('datastruct_%03i', refMaskStructIdx), sprintf('datastruct_%03i.mat', refMaskStructIdx)));
        D.refRun = ref.refRun;
        refMeta = load(ref.metaPath);
        
        
        D.refPath = ref.refPath; %Meta.file(D.refRun).si.tiffPath;
        D.slices = ref.slices;
        
        D.maskInfo = struct();
        D.maskInfo = ref.maskInfo;
        D.maskType = ref.maskType;
        
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        
    case 'pixels'

        D.slices = slicesToUse;
        
        D.maskType = 'pixels';
        D.maskInfo = struct();
        
        % Set smoothing/filtering params:
        % -----------------------------------------------------------------
        params = struct();
        params.smoothXY = true;
        params.kernelXY = 3;
        
        D.maskInfo.slices = slicesToUse; % TMP 
        D.maskInfo.params = params;
        D.maskInfo.maskType = D.maskType;
        
        % =================================================================
        % Get traces:
        % =================================================================
        tic()
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        toc();
        

    case 'cnmf'
        
        D.maskType = 'contours';
        D.slices = slicesToUse;

        % Get NMF params:
        % -----------------------------------------------------------------
        D.maskInfo = struct();
        
        params.K = 500; %50; %300; %50; %150; %35;                                      % number of components to be found
        params.tau = 1; %2; %4;                                      % std of gaussian kernel (size of neuron) 
        % tau = [1 1; 2 2];
        % K = [100; 50];

        params.p = 2;     % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
        params.merge_thr = 0.8;                                  % merging threshold
        
%         D.params.options = CNMFSetParms(...                      
%             'd1',d1,'d2',d2,...                         % dimensions of datasets
%             'search_method','dilate','dist',3,...       % search locations when updating spatial components
%             'deconv_method','constrained_foopsi',...    % activity deconvolution method
%             'temporal_iter',2,...                       % number of block-coordinate descent steps 
%             'fudge_factor',0.98,...                     % bias correction for AR coefficients
%             'merge_thr',D.params.merge_thr,...                    % merging threshold
%             'gSig',tau,...%'method','lars',... %jyr
%             'thr_method', 'nrg'... %'max'...
%             );
        
        plotoutputs = false;
        params.scaleFOV = true;
        params.removeBadFrames = false;
        
        D.maskInfo.params = params;
        D.maskInfo.maskType = D.maskType;
        D.maskInfo.slices = slicesToUse;
        
        %tic()
        [nmfoptions, D.maskInfo.maskPaths] = getRoisNMF(D, meta, plotoutputs);
        
        D.maskInfo.params.nmfoptions = nmfoptions;
        clear nmfoptions;
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        %toc()
        
        % =================================================================
        % Get traces:
        % =================================================================
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
    
    case '3Dcnmf'
        
        fstart = tic();
        
        D.maskType = '3Dcontours';
        D.slices = slicesToUse;

        %addpath(genpath('~/Repositories/ca_source_extraction'));
        %addpath(genpath('~/Repositories/NoRMCorre'));
        
        D.nmfPath = fullfile(D.datastructPath, 'nmf');
        if ~exist(D.nmfPath)
            mkdir(D.nmfPath)
        end
        
        % Specify ROI map source(s), if seeding:
        % -----------------------------------------------------------------   
        D.maskInfo = struct();
        if seedRois
            D.maskInfo.mapSource = mapSource;
            D.maskInfo.mapSlicePaths = mapSlicesPath;
            D.maskInfo.roiPath = roiPath; 
            D.maskInfo.slices = slicesToUse;
            D.maskInfo.blobType = 'difference';
            D.maskInfo.seedRois = true;
            D.maskInfo.keepAll = true;
        end

        
        % Create memmapped files and substack if needed:
        % -----------------------------------------------------------------   
        if D.average
            D.matchtrials = matchtrials;
            D.averagePath = fullfile(D.datastructPath, 'averaged');
            if ~exist(D.averagePath, 'dir')
                mkdir(D.averagePath)
            end
        end
        memmap3D(D, meta);
        fprintf('Memmapped TIFF files.\n');
        
        
        % Set NMF params for 3D pipeline:
        % -----------------------------------------------------------------   
        if D.tefo
            % NOTE: Currently, only do patch if not seeding ROIs, just based on
            % how the input spatial components are provided (i.e., inputs
            % are not parsed into patches, and don't know if original NMF
            
            params.patch_size = [15,15,5];                   % size of each patch along each dimension (optional, default: [32,32])
            params.overlap = [6,6,2];                        % amount of overlap in each dimension (optional, default: [4,4])

            params.K = 2000;                                            % number of components to be found
            params.tau = [3,3,1];                                    % std of gaussian kernel (size of neuron) 
            params.p = 2;                                            % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
            params.merge_thr = 0.8;                                  % merging threshold
        else
            params.patch_size = [32,32,8];                   % size of each patch along each dimension (optional, default: [32,32])
            params.overlap = [6,12,4];                        % amount of overlap in each dimension (optional, default: [4,4])

            params.K = 10;                                            % number of components to be found
            params.tau = [3,6,2];                                    % std of gaussian kernel (size of neuron) 
            params.p = 2;                                            % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
            params.merge_thr = 0.8;                                  % merging threshold
        end
        
        plotoutputs = false;
        %params.scaleFOV = true;
        %params.removeBadFrames = false;
        
        D.maskInfo.params = params;
        D.maskInfo.maskType = D.maskType;
        D.maskInfo.slices = slicesToUse;
        
        % TODO:  Allow for specifying nmf options out here (and just pass
        % in options to NMF, instead of setting inside getRois3Dnmf.m).
        
        
        % Run 3D CNMF pipeline:
        % -----------------------------------------------------------------   
        roistart = tic();
        
        %D.maskInfo.params.patches = true;
        
        if D.maskInfo.seedRois
            D.maskInfo.params.patches = false;
            centroids = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
            if isfield(D.maskInfo, 'blobType')
                if strcmp(D.maskInfo.blobType, 'difference')
                    seeds = centroids.DoG;
                else
                    seeds = centroids.LoG;
                end
            end
            % Add 1 to x,y bec python 0-indexes (no need to do this for
            % slice #)
            seeds(:,1) = seeds(:,1)+1;
            seeds(:,2) = seeds(:,2)+1;
            
            % Remove ignored slics:
            discardslices = find(seeds(:,3)<D.slices(1));
            seeds(discardslices,:) = [];
            seeds(:,3) = seeds(:,3) - D.slices(1) + 1; % shift so that starting slice is slice 1
            D.maskInfo.seeds = seeds;
        end
        
        
        % Run ONCE to get reference components:

        
        
%         % look at CNMF results:
%         patch_fns = dir(fullfile(D.nmfPath, '*patch_results*.mat'));
%         patch = matfile(fullfile(D.nmfPath, patch_fns(2).name));
%         nmf_fns = dir(fullfile(D.nmfPath, '*output*.mat'));
%         nmf = matfile(fullfile(D.nmfPath, nmf_fns(2).name));
%         
%         nmfoptions = nmf.options;
%         T = size(nmf.Y, 4);
%         tiffYr = reshape(nmf.Y, nmfoptions.d1*nmfoptions.d2*nmfoptions.d3, T);
%         AY = nmf.A' * tiffYr;
%         
%         plot_components_3D_GUI(nmf.Y,nmf.A,nmf.C,nmf.b,nmf.f,nmf.avgs,nmfoptions)
%         
        % =================================================================
        % Get traces:
        % =================================================================
        tracestart = tic();
        [D.tracesPath, D.nSlicesTrace] = getTraces3Dnmf(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        % Generate paths for "masks" to created in
        % getTraces3Dnmf.m:
        tmpmaskpaths = dir(fullfile(D.datastructPath, 'masks', 'nmf3D_masks*'));
        tmpmaskpaths = {tmpmaskpaths(:).name}';
        D.maskInfo.maskPaths = {};
        for tmppath=1:length(tmpmaskpaths)
            D.maskInfo.maskPaths{end+1} = fullfile(D.datastructPath, 'masks', tmpmaskpaths{tmppath});
        end
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
        fprintf('DONE:  Extracted traces!\n');
        toc(tracestart);
        
        
        
        %%
        
        % ----------
        % TEST PLOTTING w/ MW epochs:
        % ------------
        
        meta = load(D.metaPath);
        nStimuli = length(meta.condTypes);
        if ~isfield(meta, 'stimcolors')
            colors = zeros(nStimuli,3);
            for c=1:nStimuli
                colors(c,:,:) = rand(1,3);
            end
            meta.stimcolors = colors;
            save(D.metaPath, '-append', '-struct', 'meta');
        else
            colors = meta.stimcolors;
        end
        
        % Load if already created:
        nmf_fn = dir(fullfile(D.sourceDir, 'nmf_analysis', '*output*.mat'))
        tifmem_fn = dir(fullfile(D.sourceDir, sprintf('%s.mat', D.acquisitionName)));
        nmf = load(fullfile(D.sourceDir, 'nmf_analysis', nmf_fn.name));
        data = matfile(fullfile(D.sourceDir, tifmem_fn.name));
        
        % Get raw traces using spatial and temporal components of tiff Y:
        ay = mm_fun(nmf.A, data.Y);
        aa = nmf.A'*nmf.A;
        traces = ay - aa*nmf.C;
        
        tracesName = sprintf('nmftraces_Channel%02d', cidx);
        D.tracesPath = fullfile(D.datastructPath, 'traces');
        if ~exist(D.tracesPath, 'dir')
            mkdir(D.tracesPath);
        end
        % Use center of 3D roi to choose slice idx:
        center = com(A,d1,d2,d3);
        if size(center,2) == 2
            center(:,3) = 1;
        end
        center = round(center);

        %% Check out decent looking components:
        % 3, 8, 10 21 22
        tRoi = 100;
        tFile = 3;
       
        
        zplane = center(tRoi,3);
        volumeIdxs = zplane:meta.file(tFile).si.nFramesPerVolume:meta.file(tFile).si.nTotalFrames;
        tstamps = meta.file(tFile).mw.siSec(volumeIdxs);
        mwTimes = meta.file(tFile).mw.mwSec;
        
        figure();
        plot(tstamps(1:size(Y_r_out,2)),Y_r_out(tRoi,:)/Df_out(tRoi), 'k', 'linewidth',2); 
        %plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
        ylims = get(gca,'ylim');
        currRunName = meta.file(tFile).mw.runName;
        %mwCodes = meta.file(tFile).mw.pymat.(currRunName).stimIDs;
        sy = [ylims(1) ylims(1) ylims(2) ylims(2)];
        trialidx = 1;
        currStimTrialIdx = [];
        for trial=1:2:length(mwTimes)
            sx = [mwTimes(trial) mwTimes(trial+1) mwTimes(trial+1) mwTimes(trial)];
            %currStim = mwCodes(trial);
            currStim = 1;
%             if handles.stimShowAvg.Value
            patch(sx, sy, colors(currStim,:,:), 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
%             else
%                 if currStim==handles.stimMenu.Value
%                     handles.mwepochs(trial) = patch(sx, sy, colors(currStim,:,:), 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
%                     currStimTrialIdx = [currStimTrialIdx trialidx];
%                 else
%                     handles.mwepochs(trial) = patch(sx, sy, [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
%                 end
%             end
            %handles.ax4.TickDir = 'out';
            hold on;
            trialidx = trialidx + 1;
            %handles.ax4.UserData.trialEpochs = trialidx;
        end
        nEpochs = length(mwTimes);
        
        %%
    
    
        %tic()
%         [nmfoptions, D.maskInfo.maskPaths] = getRoisNMF(D, meta, plotoutputs);
%         
%         D.maskInfo.params.nmfoptions = nmfoptions;
%         clear nmfoptions;
%         save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
%         %toc()
%         
%         % =================================================================
%         % Get traces:
%         % =================================================================
%         [D.tracesPath, D.nSlicesTrace] = getTraces(D);
%         save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
end

toc()
