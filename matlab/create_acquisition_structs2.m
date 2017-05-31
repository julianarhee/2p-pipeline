clear all;
clc;

%% Set ANALYSIS PARAMETERS:

% INFO --------------------------------------------------------------------
% roitype: ['create_new', 'pixels', 'roiMap', 'cnmf', 'manual3Drois', '3Dcnmf']
% maskshape: ['circles','contours','3Dcontours','spheres']
% maskfinder: ['blobDetector', 'EMmasks', 'centroids']
% -------------------------------------------------------------------------

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

%% Set 3Dnmf params, if using roitype='3Dcnmf':

% INFO --------------------------------------------------------------------

% For TEFO:  tau=[2,2,1]
% - patches: patch_size=[15,15,15], overlap=[6,6,2], K=10
% - otherwise: K=2000

% For long 12k-res: tau=[3,6,2]
% - patches: patch_size=[32,32,], overlap=[6,12,4], K=10
% - otherwise: K=2000
% -------------------------------------------------------------------------


roiparams = struct();

fprintf('Setting ROI parameters...\n')
switch dsoptions.roitype

    case 'pixels'
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
            'spatial_method','constrained',...       % method for updating spatial components
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
roiparams.options.spatial_method = 'constrained';
roiparams.options




%% Create datastruct for current analysis:


% Create datastruct from options:
% -------------------------------------------------------------------------
fprintf('Creating new datastruct...\n')
D = create_datastruct(dsoptions);


% Set paths to input tiffs:
% -------------------------------------------------------------------------
D = set_datafile_paths(D, dsoptions);


% Get SI volume info:
% -------------------------------------------------------------------------
[D, meta] = get_meta(D, dsoptions);


% Create memmapped files & average tiffs to get "slice images":
% -------------------------------------------------------------------------
memmap3D(D, meta);


% Save analysis info:
% -------------------------------------------------------------------------
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

fprintf('Got source info for creating ROI masks.\n')


%% Create masks and get traces:

tic()

D = extract_traces_from_masks(D, meta);

toc();

fprintf('FINISHED!\n')

    
