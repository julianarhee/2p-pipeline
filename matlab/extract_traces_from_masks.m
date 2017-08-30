function D = extract_traces_from_masks(roiparams, D, meta, getref, orderROIs)

if isempty(orderROIs)
    orderROIs = false;
end

switch D.roiType
    case 'create_rois'
        D.maskType = 'circles';
        D.maskInfo.maskType = D.maskType;
        D.maskInfo = struct();
        D.maskInfo.slices = D.slices;
       
        % Choose reference:
        % -----------------------------------------------------------------
        refMeta = load(meta.metaPath);     
        if isfield(refMeta.file(1).si, 'motionRefNum')
            D.maskInfo.refRun = refMeta.file(1).si.motionRefNum;
            D.maskInfo.refPath = refMeta.file(D.refRun).si.tiffPath;
        else
            D.maskInfo.refRun = round(length(refMeta.file)/2);
            D.maskInfo.refPath = '';
        end

        % Create ROIs:
        % -----------------------------------------------------------------
        D.maskInfo.maskPaths = create_rois(D, refMeta);
        
        % Get traces with masks:
        % -----------------------------------------------------------------
        tic()
        [D.tracesPath, D.nSlicesTrace] = get_traces(D);
        toc()
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
        fprintf('Got traces from manually-created ROIs.\n');
        
    case 'roiMap'
        
        D.maskType = 'circles';
        D.maskInfo = struct();
        D.maskInfo.roiSource = D.roiSource; %roiPath; 
        D.maskInfo.slices = D.slices; %slicesToUse;
        D.maskInfo.blobType = 'difference';
       
        % Create masks from ROI maps: 
        % -----------------------------------------------------------------
        D.maskInfo.maskPaths = rois_to_masks(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
         
        % Get traces with masks:
        % -----------------------------------------------------------------
        tic()
        [D.tracesPath, D.nSlicesTrace] = get_traces(D);
        toc()
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
    
        fprintf('Got traces for roiMap analysis (%s), %s', D.name, D.acquisitionName);
        
    case 'manual3Drois'

        % Specify ROI map source(s):
        % -----------------------------------------------------------------   
        D.maskInfo = struct();
        D.maskInfo.slices = D.slices; %slicesToUse;
        D.maskInfo.roiSource = D.roiSource;
        D.maskInfo.maskfinder = D.maskfinder;
        
        % Create MASKMAT -- single mask for each trace:
        % ----------------------------------------------------------------- 

        if strcmp(D.maskType, 'spheres')
            D.maskInfo.roiPath = D.roiSource;
            roimat = load(D.maskInfo.roiSource); 
            roinames = sort(fieldnames(roimat));

            switch D.maskInfo.maskfinder
                case 'blobDetector'
                    D.maskInfo.blobType = 'difference';
                    D.maskInfo.keepAll = true;
                    
                    centroids = load(D.maskInfo.roiSource); 
                    % centroids = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
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
                    %D.maskInfo.seeds = seeds;
                    centers = seeds;
                    radii = repmat(roiparams.radius, length(centers), 1);
                    roiIDs = 1:size(centroids.DoG, 1);
 
                case 'centroids'
                    centers = zeros(length(roinames), 3);
                    radii = zeros(length(roinames), 1);
                    roiIDs = zeros(length(roinames), 1);
                    for roi=1:length(roinames)
                        centers(roi,:) = roimat.(roinames{roi}).TEFO + 1;
                        radii(roi) = roiparams.radius; %1.5;
                        roiname = roinames{roi};
                        roiIDs(roi) = str2double(roiname(5:end));
                    end
            end

            D.maskInfo.centroidsOnly = true;
            D.maskInfo.roiIDs = roiIDs;
                
            volumesize = meta.volumeSizePixels;
            %centers = round(centers);
            view_sample = false;
            maskmat = get_golfballs(centers, radii, volumesize, view_sample);
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
            volumesize = meta.volumeSizePixels;
            roimat = load(D.maskInfo.roiSource);
            roinames = sort(fieldnames(roimat));
            tmpmat = arrayfun(@(i) reshape(roimat.(roinames{i}), prod(volumesize), []), 1:length(roinames), 'UniformOutput', 0);
            maskmat = cat(2, tmpmat{1:end});
            maskstruct = struct();
            maskstruct.maskmat = maskmat;
            maskstruct.roiIDs = cellfun(@(roiname) str2double(roiname(5:end)), roinames);
            maskstuct.volumesize = volumesize;

            D.maskInfo.centroidsOnly = false;
            D.maskInfo.roiIDs = maskstruct.roiIDs;
            
            D.maskmatPath = fullfile(D.datastructPath, 'maskmat.mat');
            save(D.maskmatPath, '-struct', 'maskstruct');
        end
        
        % Get Traces for 3D masks:
        % ----------------------------------------------------------------- 
        tracestart = tic();
        [D.tracesPath, D.nSlicesTrace] = get_traces_3D(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        % Generate paths for "masks":
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
        D.maskInfo = struct();
        D.maskInfo.roiSource = D.roiSource;      
        
        % ref = load(fullfile(D.maskSourcePath, 'analysis', D.maskDatastruct, sprintf('datastruct_%03i.mat', D.maskDidx)));
        ref = load(D.maskInfo.roiSource);
        D.refRun = ref.refRun;
        refMeta = load(ref.metaPath);
         
        D.refPath = ref.refPath; %Meta.file(D.refRun).si.tiffPath;
        D.slices = ref.slices;
        
        D.maskInfo = ref.maskInfo;
        D.maskType = ref.maskType;
        D.maskInfo.maskPaths = ref.maskInfo.maskPaths;
        
        [D.tracesPath, D.nSlicesTrace] = get_traces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        
    case 'pixels'

        D.maskType = 'pixels';
        D.maskInfo = struct();
        
        % Set smoothing/filtering params:
        % -----------------------------------------------------------------       
        D.maskInfo.slices = D.slices; %slicesToUse; % TMP 
        D.maskInfo.params = roiparams;
        D.maskInfo.maskType = D.maskType;
        
        % Get traces:
        % -----------------------------------------------------------------       
        tic()
        [D.tracesPath, D.nSlicesTrace] = get_traces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        toc();
        

    case 'cnmf'
        
        D.maskType = 'contours';
       
        D.maskInfo.params = roiparams;
        D.maskInfo.maskType = D.maskType;
        D.maskInfo.slices = D.slices;
        
        % Get masks:
        % -----------------------------------------------------------------       
        [nmfoptions, D.maskInfo.maskPaths] = getRoisNMF(D, meta, plotoutputs);
        D.maskInfo.params.nmfoptions = nmfoptions;
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        % Get traces:
        % -----------------------------------------------------------------       
        [D.tracesPath, D.nSlicesTrace] = get_traces(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
    
        
    case '3Dcnmf'
        
        fstart = tic();

        D.nmfPath = fullfile(D.datastructPath, 'nmf');
        if ~exist(D.nmfPath)
            mkdir(D.nmfPath)
        end

        % Specify ROI map source(s):
        % -----------------------------------------------------------------   
        D.maskInfo = struct();
        if D.seedRois
            D.maskInfo.roiSource = D.roiSource;
            D.maskInfo.slices = D.slices;
            D.maskInfo.seedRois = true;
            D.maskInfo.maskfinder = D.maskfinder;
  
            switch D.maskInfo.maskfinder
                case 'blobDetector'
                    D.maskInfo.blobType = 'difference';
                    D.maskInfo.keepAll = true;
                    
                    centroids = load(D.maskInfo.roiSource); 
                    % centroids = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing
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
                    D.maskInfo.centroidsOnly = true;
                    D.maskInfo.roiIDs = 1:size(centroids.DoG, 1);
 
                case 'EMmasks'
                    volumesize = meta.volumeSizePixels;
                    roimat = load(D.maskInfo.roiSource); % ROI keys are slices with 1-indexing
                    roinames = sort(fieldnames(roimat));
                    tmpmat = arrayfun(@(i) reshape(roimat.(roinames{i}), prod(volumesize), []), 1:length(roinames), 'UniformOutput', 0);
                    maskmat = cat(2, tmpmat{1:end});
                    
                    D.maskInfo.centroidsOnly = false;
                    D.maskInfo.seeds = maskmat;
                    D.maskInfo.roiIDs = cellfun(@(roiname) str2double(roiname(5:end)), roinames);
                    
                case 'centroids'
                    roimat = load(D.maskInfo.roiSource); % ROI keys are slices with 1-indexing
                    roinames = sort(fieldnames(roimat));
                    fprintf('Loaded %i ROIs from file %s...\n', length(roinames), D.maskInfo.roiSource);
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
        D.maskInfo
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

 
        % Run 3D CNMF pipeline:
        % -----------------------------------------------------------------   
         
        % Run ONCE to get reference components:
        roistart = tic();
        if isempty(getref)
            getref = true;
            fprintf('Getting reference components first...\n');
        elseif ~getref
            nmffiles = dir(fullfile(D.nmfPath, 'nmfoutput_ref_*.mat'));
            if isempty(nmffiles)
                fprintf('No previous ref NMF found.  Getting new reference.\n');
                getref = true;
            else
                display(nmffiles)
                fprintf('Loading previous ref: %s\n', nmffiles(1).name);
                D.maskInfo.ref.refnmfPath = fullfile(D.nmfPath, nmffiles(1).name);
                tmpnmf = matfile(D.maskInfo.ref.refnmfPath);
                D.maskInfo.ref.nmfoptions = tmpnmf.options;
                getref = false;
                clear tmpnmf
            end
        end
        if getref
            if roiparams.patches
                D.maskInfo.params.K = roiparams.patchK;
            end
            fprintf('Order ROIs: %i\n', orderROIs);
            [nmfoptions, D.maskInfo.nmfPaths] = get_rois_3Dnmf(D, meta, roiparams.plotoutputs, getref, orderROIs);
            fprintf('Extracted components for REFERENCE tiff: File%03d!\n', D.maskInfo.ref.tiffidx)
            
            D.maskInfo.ref.refnmfPath = D.maskInfo.nmfPaths;
            D.maskInfo.ref.nmfoptions = nmfoptions;
        end
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        toc(roistart);
        
        % Run AGAIN to get other components with same spatials:
        roistart = tic();
        fprintf('Getting components from previously extracted components.\n');
        getref = false;
        orderROIs = false;
        if roiparams.patches
            % Use masks from REF, don't run patches.
            D.maskInfo.params.K = roiparams.fullK;
            D.maskInfo.patches = false;
        end
        [nmfoptions, D.maskInfo.nmfPaths] = get_rois_3Dnmf(D, meta, roiparams.plotoutputs, getref, orderROIs);
       
        D.maskInfo.params.nmfoptions = nmfoptions;
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
        fprintf('Extracted all 3D ROIs!\n')
        fprintf('**********************************\n'); 
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
        
 
        % Get traces:
        % -----------------------------------------------------------------   
        [D.tracesPath, D.nSlicesTrace] = get_traces_3Dnmf(D);
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

        % Generate paths for "masks":
        tmpmaskpaths = dir(fullfile(D.datastructPath, 'masks', 'nmf3D_masks*'));
        tmpmaskpaths = {tmpmaskpaths(:).name}';
        D.maskInfo.maskPaths = {};
        for tmppath=1:length(tmpmaskpaths)
            D.maskInfo.maskPaths{end+1} = fullfile(D.datastructPath, 'masks', tmpmaskpaths{tmppath});
        end
        save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
        
        fprintf('DONE:  TOTAL TIME for source extraction:\n');
        fprintf('**********************************\n');
        toc(fstart);


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
        
        
end
