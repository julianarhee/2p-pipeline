function [tracesPath, nSlicesTrace] = getTraces3Dnmf(D)

acquisitionName = D.acquisitionName;
nTiffs = D.nTiffs;
nchannels = D.channelIdx;
meta = load(D.metaPath);


% Create output dirs if no exist:
% -------------------------------------------------------------------------
tracesPath = fullfile(D.datastructPath, 'traces');
if ~exist(tracesPath, 'dir')
    mkdir(tracesPath);
end

masksPath = fullfile(D.datastructPath, 'masks');
if ~exist(masksPath, 'dir')
    mkdir(masksPath)
end

% Get file names of NMF results:
nmf_fns = dir(fullfile(D.nmfPath, 'nmfoutput_File*.mat'));

for fidx=1:nTiffs

    % Initialize output stucts:
    % -------------------------
    maskStruct3D = struct();
    T = struct();

    % Load NMF results:
    % -------------------------
    nmf = matfile(fullfile(D.nmfPath, nmf_fns(fidx).name)); %This is equiv. to 'maskstruct' in 2d
    
    nopts = nmf.options;
    d1=nopts.d1;
    d2=nopts.d2;
    d3=nopts.d3;
    sizY= size(nmf.Y);
    

    % Extract normalized spatial components:
    % --------------------------------------
    nA = full(sqrt(sum(nmf.A.^2))');
    [K,~] = size(nmf.C);
    spatialcomponents = nmf.A/spdiags(nA,0,K,K);


    % Get raw traces using spatial and temporal components of tiff Y:
    % ---------------------------------------------------------------
    % ay = mm_fun(nmf.A, data.Y);
    % aa = nmf.A'*nmf.A;
    % traces = ay - aa*nmf.C;
    ntpoints = sizY(end);
    traces = nmf.A' * reshape(double(nmf.Y), d1*d2*d3, ntpoints); % nRows = nROIS, nCols = tpoints
    avgs = nmf.avgs;

    nslices = size(avgs,3);

    % Create "masks" from 3D spatial components:
    % -----------------------------------------
    %roiMat = full(nmf.A); % TODO:  IS this normalized to unit energy?? check aginst "spatialcomponents" above...
    nRois = size(nmf.A,2);


    % Create cell array of sparse mats, each of which is a 3D ROI "mask".
    % ---------------------------------------------------------------------
    % MASKCELL3D - each ROI of the current FILE is stored as a sparse mat

    maskcellTmp = arrayfun(@(roi) reshape(full(nmf.A(:,roi)), d1, d2, d3), 1:nRois, 'UniformOutput', false);
    maskcell3D = cellfun(@logical, maskcellTmp, 'UniformOutput', false); % Does this work on sparse mats?
    %maskcell = cellfun(@sparse, maskcell, 'UniformOutput', false);
     
    
    % Sort ROIs by which slices they have sptial footprint on:
    % ---------------------------------------------------------------------
    roislices = cellfun(@(roimask) find(cell2mat(arrayfun(@(sl) any(find(roimask(:,:,sl))), 1:nslices, 'UniformOutput', 0))), maskcell3D, 'UniformOutput', 0);

    % Get centers of each ROI:
    centers = com(nmf.A,d1,d2,d3);
    if size(centers,2) == 2
        centers(:,3) = 1;
    end
    centers = round(centers);
    
    currSliceRois = {};
    % for sl=1:nslices
    %     currSliceRois{end+1} = find(centers(:,3)==sl);
    % end
    for sl=1:nslices
        %currSliceRois{end+1} = find(centers(:,3)==sl); % Use center of
        %mass (COM) to make 2D masks for each slice
        roi_ids_found = find(cell2mat(cellfun(@(rslices) any(find(rslices==sl)), roislices, 'UniformOutput', 0)));
        currSliceRois{end+1} = roi_ids_found; %maskstruct.roiIDs(roi_ids_found);
    end


    % TODO:  if go with 3D auto-ROIs, fix everything upstream so that we don't
    % have to have silly re-organizing of all the ROIs and masks... Much faster
    % and efficient to keep things in mat-form.

    % This is stupid because it is just reshaping everything to follow
    % structure of manually-selected ROIs to plot in roigui.m....
    tic()
    nrois = length(maskcell3D);
    %maskcell = cell(1,nrois);
    
    for sl=1:nslices

        mask_slicefn = sprintf('masks_Slice%02d.mat', D.slices(sl));
        maskPath = fullfile(D.datastructPath, 'masks', mask_slicefn);

        if exist(maskPath, 'file')
            maskStruct = load(maskPath);
        else
            maskStruct = struct();
        end

        ncurrrois = length(currSliceRois{sl});
        maskcell = cell(1,ncurrrois);
        for roi=1:length(currSliceRois{sl})%nrois
            roidx = currSliceRois{sl}(roi);
            %maskcell{roi} = maskcell3D{roidx}(:,:,centers(roidx, 3));
            maskcell{roi} = maskcell3D{roidx}(:,:,sl);
        end

        maskStruct.file(fidx).maskcell = maskcell;
        maskStruct.file(fidx).centers = centers(currSliceRois{sl},:); %currSliceRois{sl};
        if isfield(D.maskInfo, 'roiIDs')
            maskStruct.file(fidx).roi3Didxs = D.maskInfo.roiIDs(currSliceRois{sl});  %currSliceRois{sl};
        else
            maskStruct.file(fidx).roi3Didxs = currSliceRois{sl};
        end


        [fp,fn,fe] = fileparts(maskPath);

        if exist(maskPath, 'file')
            save_struct(fp, strcat(fn,fe), maskStruct, '-append')
        else
            save_struct(fp, strcat(fn,fe), maskStruct)
        end

        % Also extract traces for curr-slice masks:
        tracesName = sprintf('traces_Slice%02d_Channel%02d.mat', D.slices(sl), 1);
        if exist(fullfile(tracesPath, tracesName), 'file')
            tracestruct = load(fullfile(tracesPath, tracesName));
        else
            tracestruct = struct();
        end
        tracestruct.file(fidx).rawTraces = traces(currSliceRois{sl},:)'; %rawTraces;

        % Include CNMF output to struct:

        dfTracesNMF = nmf.F_df;
        dfTracesNMF = dfTracesNMF{1};
        detrendedNMF = nmf.Fd_us;
        detrendedNMF = detrendedNMF{1};

        rawNMF = nmf.A*nmf.C + nmf.b*nmf.f;
        rawTracesNMF = nmf.A'*rawNMF;
        
%         currDfs = tmpDf(currSliceRois{sl});
%         inferredTraces = arrayfun(@(i) tmpC(currSliceRois{sl}(i),:)/currDfs(i), 1:ncurrrois, 'UniformOutput', 0);
%         inferredTraces = cat(1, inferredTraces{1:end});

%         tracestruct.file(fidx).inferredTraces = inferredTraces'; 
        tracestruct.file(fidx).rawTracesNMF = rawTracesNMF(currSliceRois{sl},:)';
        tracestruct.file(fidx).dfTracesNMF = dfTracesNMF(currSliceRois{sl},:)';
        tracestruct.file(fidx).detrendedTracesNMF = detrendedNMF(currSliceRois{sl},:)';
        %T.masks.file(fidx) = {masks};
        tracestruct.file(fidx).avgImage = avgs(:,:,sl);
        %T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
        %T.file(fidx).badFrames = badframes;
        %T.file(fidx).corrcoeffs = corrs;
        %T.file(fidx).refframe = refframe;
        tracestruct.file(fidx).info.szFrame = sizY(1:2); %size(avgY);
        tracestruct.file(fidx).info.nFrames = sizY(:,end);
        tracestruct.file(fidx).info.nRois = length(currSliceRois{sl}); %size(tr,1); %length(maskcell);

        if exist(fullfile(tracesPath, tracesName), 'file')
            save_struct(tracesPath, tracesName, tracestruct, '-append');
        else
            fprintf('Saving struct: %s.\n', tracesName);
            save_struct(tracesPath, tracesName, tracestruct);
        end

    end

    
    % Save 3D mask info for current movie:
    % ------------------------------------
    maskStruct3D.maskcell3D = maskcell3D;
    maskStruct3D.centers = centers;
    maskStruct3D.spatialcomponents = spatialcomponents;
    maskStruct3D.roiMat = full(nmf.A); %roiMat;
    if ~isfield(D.maskInfo, 'roiIDs')
        maskStruct3D.roi3Didxs = 1:nrois;
    else
        maskStruct3D.roi3Didxs = D.maskInfo.roiIDs; %maskstruct.roiIDs;
    end

    % This maskPath points to the 3D mask cell array, i.e., each cell contains
    % 3D mask, and length of cell array is nRois:
    maskpath3D = fullfile(D.datastructPath, 'masks', sprintf('nmf3D_masks_File%03d.mat', fidx))

    [fp,fn,fe] = fileparts(maskpath3D);
    %save_struct(fp, strcat(fn,fe), maskStruct3D)
    save(maskpath3D, '-struct', 'maskStruct3D','-v7.3');


    fprintf('Saving struct: %s.\n', maskpath3D);

    %else

    % Save 3D traces info for current movie:
    % ------------------------------------
    T.rawTraces = traces'; %rawTraces;

    % Include nmf outptu infered traces:
    %inferredTraces = arrayfun(@(i) tmpC(i,:)/tmpDf(i), 1:nrois, 'UniformOutput', 0);
    %inferredTraces = cat(1, inferredTraces{1:end});
    %T.inferredTraces = inferredTraces';
    T.rawTracesNMF = rawTracesNMF';
    T.dfTracesNMF = dfTracesNMF';
    T.detrendedNMF = detrendedNMF';

    %T.masks.file(fidx) = {masks};
    T.avgImage = avgs;
    %T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
    %T.file(fidx).badFrames = badframes;
    %T.file(fidx).corrcoeffs = corrs;
    %T.file(fidx).refframe = refframe;
    T.info.szFrame = sizY(1:2); %size(avgY);
    T.info.nFrames = sizY(end);
    T.info.nRois = length(maskStruct3D); %length(maskcell);

    %clearvars rawTraces rawTracesTmp cellY Y avgY corrs

    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);

    % Save traces for each file to slice struct:                    
    %                 tracesPath = fullfile(dstructPath, 'traces');
    %                 if ~exist(tracesPath, 'dir')
    %                     mkdir(tracesPath);
    %                 end
    tracesName = sprintf('traces3D_Channel%02d_File%03d', 1, fidx);
    fprintf('Saving struct: %s.\n', tracesName);

    save_struct(tracesPath, tracesName, T);


    % 
    tracesSaved = dir(fullfile(tracesPath, 'traces_Slice*'));
    tracesSaved = tracesSaved(arrayfun(@(x) ~strcmp(x.name(1),'.'),tracesSaved));
    nSlicesTrace = length(tracesSaved);




end
