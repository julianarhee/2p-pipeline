clear all;
clc;

%% DEFINE SOURCE DIRECTORY:


% -------------------------------------------------------------------------
% retinotopyFinal/
% -------------------------------------------------------------------------
%sourceDir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal';
% acquisitionName = 'fov6_retinobar_037Hz_final_nomask';
% extraTiffsExcluded = [];
% tefo = true;
% 
% average = true;
% matchtrials = [{[1, 1]}, {[2, 3]}, {[3, 2]}, {[4, 4]}];
% 
% dsoptions = DSoptions(...
%     'source', '/nas/volume1/2photon/RESDATA/TEFO',...           % parent dir
%     'session', '20161219_JR030W',...                            % session name (single FOV)
%     'run', 'retinotopyFinal',...                                % experiment name
%     'datastruct', 18,...                                        % datastruct idx
%     'acquisition', 'fov6_retinobar_037Hz_final_nomask',...      % acquisition name
%     'datapath', 'DATA',...       % alternate datapath if preprocessed (default: '')
%     'tefo', true,...                                            % 'scope type' (t/f)
%     'preprocessing', 'raw',...                                  % preprocessed or no
%     'meta', 'SI',...                                            % source of meta info
%     'channels', 2,...                                           % num channels acquired
%     'signalchannel', 1,...                                      % channel num of signal
%     'roitype', '3Dcnmf',...                                     % method for roi extraction
%     'seedrois', 'true',...                                      % provide external source of seed coords
%     'maskdims', '3D',...                                        % dimensions of masks
%     'maskshape', '3Dcontours',...                               % shape of masks
%     'maskfinder', 'EMmasks',...                                 % method of finding masks, given set of seed coords
%     'slices', [9:30],...                                        % slices from acquis. that actually contain data
%     'averaged', true,...                                        % using tiffs that are the averaged tcourses of runs
%     'excludedtiffs', [],...                                     % idxs of tiffs to exclude from analysis
%     'metaonly', false,...                                       % only get meta data from tiffs (if files too large)
%     'nmetatiffs', 4);                                           % number of tiffs need meta-only info for (default: 0)
% 


%% TEST NEW:

dsoptions = DSoptions(...
    'source', '/nas/volume1/2photon/RESDATA/TEFO',...           % parent dir
    'session', '20161218_CE024',...                            % session name (single FOV)
    'run', 'retinotopy5',...                                % experiment name
    'datastruct', 1,...                                        % datastruct idx
    'acquisition', 'fov2_bar5',...      % acquisition name
    'datapath', 'DATA',...          % preprocessed datapath 
    'tefo', true,...                                            % 'scope type' (t/f)
    'preprocessing', 'raw',...                                  % preprocessed or no
    'corrected', false,...                                      % corrected (w/ Acq2P or no)
    'meta', 'SI',...                                            % source of meta info
    'channels', 2,...                                           % num channels acquired
    'signalchannel', 1,...                                      % channel num of signal
    'roitype', '3Dcnmf',...                                     % method for roi extraction
    'seedrois', false,...                                      % provide external source of seed coords
    'maskdims', '3D',...                                        % dimensions of masks
    'maskshape', '3Dcontours',...                               % shape of masks
    'maskfinder', '',...                                 % method of finding masks, given set of seed coords
    'slices', [1:12],...                                        % slices from acquis. that actually contain data
    'averaged', false,...                                        % using tiffs that are the averaged tcourses of runs
    'matchedtiffs', [],...                                      % matched tiffs, if averaging
    'excludedtiffs', [],...                                     % idxs of tiffs to exclude from analysis
    'metaonly', true,...                                       % only get meta data from tiffs (if files too large)
    'nmetatiffs', 4);                                           % number of huge tiffs to exclude

%%
% ---------------------------------------------------------------
% Set mask source paths if seeding ROIs:
% ----------------------------------------------------------------
if dsoptions.seedrois
    fprintf('Getting info on seed ROIs...\n');
    [fpath,fcond,~] = fileparts(D.sourceDir);
    mapdir = 'em7_centroids'; 
    if strcmp(dfsoptions.roitype, 'condition')
        % Also need to specify which datastruct no. to get masks from:
        refidx = 1;
    end
    mapSource = fullfile(fpath, mapdir); % fullfile(fpath, 'average_volumes');
    centroidfnpattern = 'rois*.mat';
    maskfnpattern = 'masks*.mat';
    mapSlicesPath = fullfile(mapSource, 'average_stacks', 'Channel01'); % Path to slices onto which masks can be drawn, ex: fullfile(fpath, 'avg_frames_conditions_channel01');
end


% -----------------------------------------------------------------   
% Set 3Dnmf params, if using roitype='3Dcnmf':
% -----------------------------------------------------------------   
% For TEFO:  tau=[2,2,1]
% - patches: patch_size=[15,15,15], overlap=[6,6,2], K=10
% - otherwise: K=2000
% For long 12k-res: tau=[3,6,2]
% - patches: patch_size=[32,32,], overlap=[6,12,4], K=10
% - otherwise: K=2000
% -----------------------------------------------------------------

fprintf('Setting ROI parameters...\n')
switch dsoptions.roitype

    case 'pixels'
        roiparams = struct();
        roiparams.smoothXY = true;
        roiparams.kernelXY = 3;
 
    case '3Dcnmf'
        roiparams.refidx = 2;                       % tiff idx to use as reference for spatial components
        roiparams.tau = [2,2,1];                    % std of gaussian kernel (size of neuron) 
        roiparams.p = 2;                            % order (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
        roiparams.merge_thr = 0.8;                  % merging threshold
     
        if dsoptions.seedrois
            roiparams.patches = false;
        else
            roiparams.patches = false;
        end
       
        if roiparams.patches
            roiparams.K = 10;                        % number of components to be found
            roiparams.patch_size = [15,15,5];        % size of each patch along each dimension (optional, default: [32,32])
            roiparams.overlap = [6,6,2];             % amount of overlap in each dimension (optional, default: [4,4])
        else
            roiparams.K = 2000;
        end

        roiparams.plotoutputs = false;
        %params.scaleFOV = true;
        %params.removeBadFrames = false;


        roiparams.options = CNMFSetParms(...
            'spatial_method', 'constrained',...
            'search_method','ellipse','dist',2,...
            'se', strel('disk', 1, 0),...      
            'max_size', 3, 'min_size', 1,...         % max/min size of ellipse axis (default: 8, 3)
            'deconv_method','constrained_foopsi',... % activity deconvolution method
            'temporal_iter',2,...                    % number of block-coordinate descent steps 
            'cluster_pixels',false,...                  
            'ssub',1,...                             % spatial downsampling when processing
            'tsub',1,...                             % further temporal downsampling when processing
            'fudge_factor',0.96,...                  % bias correction for AR coefficients
            'merge_thr', roiparams.merge_thr,...                   % merging threshold
            'gSig',roiparams.tau,... 
            'max_size_thr',3,'min_size_thr',1,...    % max/min acceptable size for each component
            'spatial_method','regularized',...       % method for updating spatial components ('constrained')
            'df_prctile',50,...                      % take the median of background fluorescence to compute baseline fluorescence 
            'time_thresh',0.6,...
            'space_thresh',0.6,...
            'thr_method', 'max',...                  % method to threshold ('max' or 'nrg', default 'max')
            'maxthr', 0.05,...                       % threshold of max value below which values are discarded (default: 0.1)
            'conn_comp', false);                     % extract largest connected component (binary, default: true)

end

% -----------------------------------------------------------------   
% Create datastruct for analysis:
% -----------------------------------------------------------------   
fprintf('Creating new datastruct...\n')
D = create_datastruct(dsoptions);

% --------------------------------------------
% 3.  Specify ROI/mask params/paths:
% --------------------------------------------
D = set_mask_paths(D, dsoptions);

% TODO:  create txt file with datastruct info.
%dstructFile = 'datastructs.txt'
%dstructPath = fullfile(source, session, run, dstructFile)
%headers = {'datastruct', 'acquisition', 'tefo', 'preprocessing', 'meta',...
%            'roiType', 'seedrois', 'maskdims', 'maskshape', 'maskfinder',...
%            'channels', 'signalchannel', 'slices', 'averaged',...
%             'excludedtiffs', 'metaonly', 'hugetiffs'};
%
%fid = fopen(dstructPath, 'w');
%if ~exist(dstructPath)
%    fprintf(fid, '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\r\n', headers{1:end});
%else
%    fprintf(fid, '%s\t%s\t%i\t%s\%s\t%s\t%i\t%s\t%s\t%s\t%i\t%i\t%s\t%i\t%s\t%s\t%i\t%i', dstructoptions{1:end});
%    fclose(fid);
%end
%


% Set paths to input tiffs:
% -----------------------------------------------------------------  
D = set_datafile_paths(D, dsoptions);



%% Get SI volume info:
[D, meta] = get_meta(D, dsoptions);


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

switch D.roiType
    case 'create_rois'
        
        % Choose reference:
        % -----------------------------------------------------------------
        refMeta = load(meta.metaPath);     
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
        D.slices = slicesToUse; 
       
 
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
        
        % D.maskType = manual3Dshape;  %TODO:  FIX THSI in datastruct_011 -- should be "spheres" %'3Dcontours';

       
        % Create memmapped files and substack if needed:
        % -----------------------------------------------------------------          
        memmap3D(D, meta); %, create_substack);
        fprintf('Memmapped TIFF files.\n');
        
        
        % Specify ROI map source(s), if seeding:
        % -----------------------------------------------------------------   
        D.maskInfo = struct();

        D.maskInfo.mapSource = mapSource;  % path to AVG volumes from which ROIs selected
        D.maskInfo.mapSlicePaths = mapSlicesPath;  % path to each slice of AVG volume (mapSource is parent)
         
        D.maskInfo.slices = D.slices; %slicesToUse;

        % Create MASKMAT -- single mask for each trace:
        % ----------------------------------------------------------------- 
        if strcmp(D.maskType, 'spheres')
            D.maskInfo.roiPath = roiCentroidPath;
            
            roimat = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
            roinames = sort(fieldnames(roimat));

            centers = zeros(length(roinames), 3);
            radii = zeros(length(roinames), 1);
            roiIDs = zeros(length(roinames), 1);
            for roi=1:length(roinames)
                centers(roi,:) = roimat.(roinames{roi}).TEFO + 1;
                radii(roi) = 1.5;
                roiname = roinames{roi};
                roiIDs(roi) = str2double(roiname(5:end));
            end
        
            volumesize = meta.volumeSizePixels;
            %centers = round(centers);
            view_sample = false;
            maskmat = getGolfballs(centers, radii, volumesize, view_sample);
            D.maskmatPath = fullfile(D.datastructPath, 'maskmat.mat');
            maskstruct = struct();
            maskstruct.centroids = centers;
            maskstruct.radii = radii;
            maskstruct.maskmat = maskmat;
            maskstruct.volumesize = volumesize;
            maskstruct.roiIDs = roiIDs;
            save(D.maskmatPath, '-struct', 'maskstruct');
        else
            % USE ACTUAL MASKS:
            % LOAD MASK
            volumesize = meta.volumeSizePixels;
            D.maskInfo.roiPath = roiMaskPath;
            roimat = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
            roinames = sort(fieldnames(roimat));
            tmpmat = arrayfun(@(i) reshape(roimat.(roinames{i}), prod(volumesize), []), 1:length(roinames), 'UniformOutput', 0);
            maskmat = cat(2, tmpmat{1:end});
            maskstruct = struct();
            maskstruct.maskmat = maskmat;
            maskstruct.roiIDs = cellfun(@(roiname) str2double(roiname(5:end)), roinames);
            maskstuct.volumesize = volumesize;
            D.maskmatPath = fullfile(D.datastructPath, 'maskmat.mat');
            save(D.maskmatPath, '-struct', 'maskstruct');
        end
        
        % Get Traces for 3D masks:
        % ----------------------------------------------------------------- 
        tracestart = tic();
        [D.tracesPath, D.nSlicesTrace] = getTraces3D(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        % Generate paths for "masks" to created in
        % getTraces3Dnmf.m:
        % ----------------------------------------------------------------- 
        tmpmaskpaths = dir(fullfile(D.datastructPath, 'masks', 'manual3D_masks*'));
        tmpmaskpaths = {tmpmaskpaths(:).name}';
        D.maskInfo.maskPaths = {};
        for tmppath=1:length(tmpmaskpaths)
            D.maskInfo.maskPaths{end+1} = fullfile(D.datastructPath, 'masks', tmpmaskpaths{tmppath});
        end
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
        fprintf('DONE:  Extracted traces!\n');
        toc(tracestart);

       
    case 'condition'
        % Use masks from a different condition:
        ref = load(fullfile(D.maskSourcePath, 'analysis', D.maskDatastruct, sprintf('datastruct_%03i.mat', D.maskDidx)));
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

        D.maskType = 'pixels';
        D.maskInfo = struct();
        
        % Set smoothing/filtering params:
        % -----------------------------------------------------------------       
        D.maskInfo.slices = D.slices; %slicesToUse; % TMP 
        D.maskInfo.params = roiparams;
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
       
        D.maskInfo.params = roiparams;
        D.maskInfo.maskType = D.maskType;
        D.maskInfo.slices = D.slices;
        
        %tic()
        [nmfoptions, D.maskInfo.maskPaths] = getRoisNMF(D, meta, plotoutputs);
        
        D.maskInfo.params.nmfoptions = nmfoptions;
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        %toc()
        
        % =================================================================
        % Get traces:
        % =================================================================
        [D.tracesPath, D.nSlicesTrace] = getTraces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
    
    case '3Dcnmf'
        
        fstart = tic();
        
        % D.maskType = manual3Dshape; %'3Dcontours';
       
        D.nmfPath = fullfile(D.datastructPath, 'nmf');
        if ~exist(D.nmfPath)
            mkdir(D.nmfPath)
        end

        % Create memmapped files and substack if needed:
        % -----------------------------------------------------------------   
        memmap3D(D, meta); %, create_substack);
        fprintf('Memmapped TIFF files.\n');
        
        
        % Specify ROI map source(s), if seeding:
        % -----------------------------------------------------------------   
        D.maskInfo = struct();
        if D.seedRois
            D.maskInfo.mapSource = mapSource;
            D.maskInfo.mapSlicePaths = mapSlicesPath;
            D.maskInfo.slices = D.slices;
            D.maskInfo.seedRois = true;
            
            switch maskFinder
                case 'blobDetector'
                    D.maskInfo.roiPath = roiPath; 
                    D.maskInfo.blobType = 'difference';
                    D.maskInfo.keepAll = true;
                    
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
                    % NOTE:  05/24/2017 -- roi_blob_detector.ipynb saves output of [y,x,r] for slices as:
                    % [x,y,z+1] -- don't need to +1 for z, and don't need to swap x,y.
                    seeds(:,1) = seeds(:,1)+1;
                    seeds(:,2) = seeds(:,2)+1;

                    % Remove ignored slics:
                    discardslices = find(seeds(:,3)<D.slices(1));
                    seeds(discardslices,:) = [];
                    seeds(:,3) = seeds(:,3) - D.slices(1) + 1; % shift so that starting slice is slice 1
                    D.maskInfo.seeds = seeds;
                    
                case 'EMmasks'
                    volumesize = meta.volumeSizePixels;
                    D.maskInfo.roiPath = roiMaskPath;
                    roimat = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
                    roinames = sort(fieldnames(roimat));
                    tmpmat = arrayfun(@(i) reshape(roimat.(roinames{i}), prod(volumesize), []), 1:length(roinames), 'UniformOutput', 0);
                    maskmat = cat(2, tmpmat{1:end});
                    
                    D.maskInfo.centroidsOnly = false;
                    D.maskInfo.seeds = maskmat;
                    D.maskInfo.roiIDs = cellfun(@(roiname) str2double(roiname(5:end)), roinames);
                    
                case 'centroids'
                    D.maskInfo.roiPath = roiCentroidPath;

                    roimat = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
                    roinames = sort(fieldnames(roimat));
                    fprintf('Loaded %i ROIs from file %s...\n', length(roinames), D.maskInfo.roiPath);
                    centers = zeros(length(roinames), 3);
                    radii = zeros(length(roinames), 1);
                    roiIDs = zeros(length(roinames), 1);
                    for roi=1:length(roinames)
                        centers(roi,:) = roimat.(roinames{roi}).TEFO + 1; % +1 bec python 0-indexing
                        radii(roi) = 1.5;
                        roiname = roinames{roi};
                        roiIDs(roi) = str2double(roiname(5:end));
                    end
                    
                    D.maskInfo.seeds = centers;
                    D.maskInfo.seeds(:,1) = centers(:,2); % Swap x,y so that i,j = y,x in image...
                    D.maskInfo.seeds(:,2) = centers(:,1);
                    D.maskInfo.roiIDs = roiIDs;
                    D.maskInfo.keepAll = false; %true;
                    D.maskInfo.centroidsOnly = true;
                    fprintf('Seeding %i ROIs', length(centers));
                    
            end
        else
            D.maskInfo.slices = D.slices;
            D.maskInfo.seedRois = false;
            D.maskInfo.keepAll = false;
        end
        
        % Add volume size info:
        roiparams.options.d1 = meta.volumeSizePixels(1);
        roiparams.options.d2 = meta.volumeSizePixels(2);
        roiparams.options.d3 = meta.volumeSizePixels(3);        
 
        D.maskInfo.params = roiparams;
        D.maskInfo.nmfoptions = roiparams.options;
        D.maskInfo.patches = roiparams.patches;
        D.maskInfo.maskType = D.maskType;
        D.maskInfo.slices = D.slices;
        D.maskInfo.ref.tiffidx = roiparams.refidx;
             
        % Run 3D CNMF pipeline:
        % -----------------------------------------------------------------   
         
        % Run ONCE to get reference components:
        roistart = tic();

        getref = true;
        [nmfoptions, D.maskInfo.nmfPaths] = getRois3Dnmf(D, meta, roiparams.plotoutputs, getref);
        fprintf('Extracted components for REFERENCE tiff: File%03d!\n', D.maskInfo.ref.tiffidx)
        
        D.maskInfo.ref.refnmfPath = D.maskInfo.nmfPaths;
        D.maskInfo.ref.nmfoptions = nmfoptions;
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        toc(roistart);
        
        % Run AGAIN to get other components with same spatials:
        getref = false;
        [nmfoptions, D.maskInfo.nmfPaths] = getRois3Dnmf(D, meta, roiparams.plotoutputs, getref);
       
        D.maskInfo.params.nmfoptions = nmfoptions;
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
        fprintf('Extracted all 3D ROIs!\n')
        
        toc(roistart);
        
        
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


%        if make_videos
%            for nmfpathdx = 1:length(D.maskInfo.nmfPaths)
%                nmf = matfile(D.maskInfo.maskPaths{nmfpathidx});
%                [A_or,C_or,S_or,P_or] = order_ROIs(nmf.A,nmf.C,nmf.S,nmf.P); % order components
%                nmfoptions.ind = [1:10];        % indices of components to be shown
%                nmfoptions.mak_avi = 1;         % flag for saving avi video (default: 0)
%                nmfoptions.show_contours = 1;   % flag for showing contour plots of patches in FoV (default: 0)
%                movname = sprintf('Movie_components_File%03d.avi', nmfpathidx)
%                nmfoptions.name = fullfile(D.nmfPath, movname);
%                [Coords, jsonCoords] = plot_contours(A_or, ... 
%                make_patch_video(A_or, C_or, nmf.b, nmf.f, nmf.Yr, Coords, nmfoptions);


end        
        
        
    
