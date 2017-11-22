function [tracesPath, nSlicesTrace] = get_traces_3Dnmf(D)

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
nmf_fns = dir(fullfile(D.nmfPath, 'nmfoutput_*.mat'));


for fidx=1:nTiffs

   % Initialize output stucts:
   % -------------------------
    if D.memmapped
        % Create memmapped files:
        tracesName = sprintf('traces3D_Channel%02d_File%03d.mat', 1, fidx);
        T = matfile(fullfile(tracesPath, tracesName), 'Writable', true);
        maskstruct3D = matfile(fullfile(D.datastructPath, 'masks', sprintf('nmf3D_masks_File%03d.mat', fidx)), 'Writable', true);
    else
        maskstruct3D = struct();
        T = struct();
    end

    % Load NMF results:
    % -------------------------
    display(fidx); nmf = matfile(fullfile(D.nmfPath, nmf_fns(fidx).name)); %This is equiv. to 'maskstruct' in 2d
    
    nopts = nmf.options;
    d1=nopts.d1;
    d2=nopts.d2;
    d3=nopts.d3;
    sizY= size(nmf.Y);
    

    % Extract normalized spatial components:
    % --------------------------------------
    nA = full(sqrt(sum(nmf.A.^2))');
    [K,~] = size(nmf.C);
    maskstruct3D.spatialcomponents = nmf.A/spdiags(nA,0,K,K); clear spatialcomponents;
    maskstruct3D.roiMat = nmf.A; %full(nmf.A); %roiMat;


    % Get raw traces using spatial and temporal components of tiff Y:
    % ---------------------------------------------------------------
    % ay = mm_fun(nmf.A, data.Y);
    % aa = nmf.A'*nmf.A;
    % traces = ay - aa*nmf.C;
    ntpoints = sizY(end);
    traces = logical(nmf.A)' * reshape(double(nmf.Y), d1*d2*d3, ntpoints); % nRows = nROIS, nCols = tpoints 
    T.rawTraces = traces'; %rawTraces; 
    clear traces;
  
    rawTracesNMF = logical(nmf.A)'*(nmf.A*nmf.C + nmf.b*nmf.f);
    T.rawTracesNMF = rawTracesNMF'; clear rawTracesNMF;
    %avgs = nmf.avgs;

    nslices = size(nmf.avgs,3);
    
    % Include CNMF output to struct:
    dfTracesNMF = nmf.F_df;
    dfTracesNMF = dfTracesNMF{1};
    detrendedNMF = nmf.Fd_us;
    detrendedNMF = detrendedNMF{1};

    %T.rawTracesNMF = rawTracesNMF';
    T.dfTracesNMF = dfTracesNMF'; clear dfTracesNMF
    T.detrendedNMF = detrendedNMF'; clear detrendedNMF



    % Create "masks" from 3D spatial components:
    % -----------------------------------------
    %roiMat = full(nmf.A); % TODO:  IS this normalized to unit energy?? check aginst "spatialcomponents" above...
    nrois = size(nmf.A,2);


    % Create cell array of sparse mats, each of which is a 3D ROI "mask".
    % ---------------------------------------------------------------------
    % MASKCELL3D - each ROI of the current FILE is stored as a sparse mat

    maskcellTmp = arrayfun(@(roi) reshape(full(nmf.A(:,roi)), d1, d2, d3), 1:nrois, 'UniformOutput', false);
    maskcell3D = cellfun(@logical, maskcellTmp, 'UniformOutput', false); % Does this work on sparse mats?
    %maskcell = cellfun(@sparse, maskcell, 'UniformOutput', false);
    clear maskcellTmp;     
    
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
    nrois = length(maskcell3D)
    %maskcell = cell(1,nrois);
    
    for sl=1:nslices

        mask_slicefn = sprintf('masks_Slice%02d.mat', D.slices(sl));
        maskPath = fullfile(D.datastructPath, 'masks', mask_slicefn);

        if exist(maskPath, 'file')
            maskstruct = load(maskPath);
        else
            maskstruct = struct();
        end

        ncurrrois = length(currSliceRois{sl});
        maskcell = cell(1,ncurrrois);
        for roi=1:length(currSliceRois{sl})%nrois
            roidx = currSliceRois{sl}(roi);
            %maskcell{roi} = maskcell3D{roidx}(:,:,centers(roidx, 3));
            maskcell{roi} = maskcell3D{roidx}(:,:,sl);
        end

        maskstruct.file(fidx).maskcell = maskcell; clear maskcell;
        maskstruct.file(fidx).centers = centers(currSliceRois{sl},:); %currSliceRois{sl};
        if isfield(D.maskInfo, 'roiIDs')
            maskstruct.file(fidx).roi3Didxs = D.maskInfo.roiIDs(currSliceRois{sl});  %currSliceRois{sl};
        else
            maskstruct.file(fidx).roi3Didxs = currSliceRois{sl};
        end


        [fp,fn,fe] = fileparts(maskPath);

        if exist(maskPath, 'file')
            %save_struct(fp, strcat(fn,fe), maskstruct, '-append')
            save(maskPath, '-append', '-struct', 'maskstruct', '-v7.3');
	else
	    %save_struct(fp, strcat(fn,fe), maskstruct)
      	    save(maskPath,'-struct', 'maskstruct', '-v7.3');
        end

        % Also extract traces for curr-slice masks:
        tracesName = sprintf('traces_Slice%02d_Channel%02d.mat', D.slices(sl), 1);
        if exist(fullfile(tracesPath, tracesName), 'file')
            tracestruct = load(fullfile(tracesPath, tracesName));
        else
            tracestruct = struct();
        end
        tmptraces = T.rawTraces;
        tracestruct.file(fidx).rawTraces = tmptraces(:,currSliceRois{sl}); clear tmptraces;%rawTraces;
        tmprawtraces = T.rawTracesNMF;
        tracestruct.file(fidx).rawTracesNMF = tmprawtraces(:,currSliceRois{sl}); clear tmprawtraces; %T.rawTracesNMF(:,currSliceRois{sl}); %rawTracesNMF(currSliceRois{sl},:)';

        dftraces = T.dfTracesNMF;
        tracestruct.file(fidx).dfTracesNMF = dftraces(:,currSliceRois{sl}); clear dftraces;
        detrendtraces = T.detrendedNMF;
        tracestruct.file(fidx).detrendedTracesNMF = detrendtraces(:,currSliceRois{sl}); clear detrendtraces; %T.detrendedNMF(:,currSliceRois{sl}); %detrendedNMF(currSliceRois{sl},:)';
        %clearvars dfTracesNMF detrendedNMF

        %rawTracesNMF = nmf.A*nmf.C + nmf.b*nmf.f;
        %rawTracesNMF = nmf.A'*rawNMF;
        
%         currDfs = tmpDf(currSliceRois{sl});
%         inferredTraces = arrayfun(@(i) tmpC(currSliceRois{sl}(i),:)/currDfs(i), 1:ncurrrois, 'UniformOutput', 0);
%         inferredTraces = cat(1, inferredTraces{1:end});

%         tracestruct.file(fidx).inferredTraces = inferredTraces'; 
        %tracestruct.file(fidx).rawTracesNMF = T.rawTracesNMF(:,currSliceRois{sl}); %rawTracesNMF(currSliceRois{sl},:)';
        %T.masks.file(fidx) = {masks};
        tracestruct.file(fidx).avgImage = nmf.avgs(:,:,sl);
        %T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
        %T.file(fidx).badFrames = badframes;
        %T.file(fidx).corrcoeffs = corrs;
        %T.file(fidx).refframe = refframe;
        tracestruct.file(fidx).info.szFrame = sizY(1:2); %size(avgY);
        tracestruct.file(fidx).info.nFrames = sizY(:,end);
        tracestruct.file(fidx).info.nRois = length(currSliceRois{sl}); %size(tr,1); %length(maskcell);

        if exist(fullfile(tracesPath, tracesName), 'file')
            save(fullfile(tracesPath, tracesName), '-append', '-struct', 'tracestruct', '-v7.3');
	    %save_struct(tracesPath, tracesName, tracestruct, '-append');
        else
            fprintf('Saving struct: %s.\n', tracesName);
            save(fullfile(tracesPath, tracesName), '-struct', 'tracestruct', '-v7.3');
	    %save_struct(tracesPath, tracesName, tracestruct);
        end
        
    end

    
    % Save 3D mask info for current movie:
    % ------------------------------------
    maskstruct3D.maskcell3D = maskcell3D; clear maskcell3D;
    maskstruct3D.centers = centers; clear centers;
    %maskstruct3D.spatialcomponents = spatialcomponents; clear spatialcomponents;
    %maskstruct3D.roiMat = nmf.A; %full(nmf.A); %roiMat;
    if ~isfield(D.maskInfo, 'roiIDs')
        maskstruct3D.roi3Didxs = 1:nrois;
    else
        maskstruct3D.roi3Didxs = D.maskInfo.roiIDs; %maskstruct.roiIDs;
    end

    % This maskPath points to the 3D mask cell array, i.e., each cell contains
    % 3D mask, and length of cell array is nRois:
    % maskpath3D = fullfile(D.datastructPath, 'masks', sprintf('nmf3D_masks_File%03d.mat', fidx))

    % [fp,fn,fe] = fileparts(maskpath3D);
    %save_struct(fp, strcat(fn,fe), maskstruct3D)
    % save(maskpath3D, '-struct', 'maskstruct3D','-v7.3');


    %fprintf('Saving struct: %s.\n', maskpath3D);

    %else

    % Save 3D traces info for current movie:
    % ------------------------------------
    %T.rawTraces = traces'; %rawTraces;

    % Include nmf outptu infered traces:
    %inferredTraces = arrayfun(@(i) tmpC(i,:)/tmpDf(i), 1:nrois, 'UniformOutput', 0);
    %inferredTraces = cat(1, inferredTraces{1:end});
    %T.inferredTraces = inferredTraces';
    %T.rawTracesNMF = rawTracesNMF';
    %T.dfTracesNMF = dfTracesNMF';
    %T.detrendedNMF = detrendedNMF';

    %T.masks.file(fidx) = {masks};
    T.avgImage = nmf.avgs;
    %T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
    %T.file(fidx).badFrames = badframes;
    %T.file(fidx).corrcoeffs = corrs;
    %T.file(fidx).refframe = refframe;
    %T.info = struct();
    info.szFrame = sizY(1:2); %size(avgY);
    info.nFrames = sizY(end);
    info.nRois = nrois; %length(maskstruct3D); %length(maskcell);
    T.info = info;

    %clearvars rawTraces rawTracesTmp cellY Y avgY corrs

    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);

    % Save traces for each file to slice struct:                    
    %                 tracesPath = fullfile(dstructPath, 'traces');
    %                 if ~exist(tracesPath, 'dir')
    %                     mkdir(tracesPath);
    %                 end
    %tracesName = sprintf('traces3D_Channel%02d_File%03d', 1, fidx);
    %fprintf('Saving 3D struct: %s.\n', tracesName);

    %save_struct(tracesPath, tracesName, T);


    % 
    tracesSaved = dir(fullfile(tracesPath, 'traces_Slice*'));
    tracesSaved = tracesSaved(arrayfun(@(x) ~strcmp(x.name(1),'.'),tracesSaved));
    nSlicesTrace = length(tracesSaved);

    clearvars tracestruct maskstruct %traces dfTracesNMF detrendedNMF rawTracesNMF 


end
