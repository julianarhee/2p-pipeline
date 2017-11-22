function [tracesPath, nSlicesTrace] = get_traces(D)

%                                     
% CASES:
%
% acquisition2p :  
%  This uses the motion-corrected & output formating used by the
%  HarveyLab Repo Acquistion2p_class with julianarhee's fork of it.
%  Slices and Channels are parsed out, with each frame saved as an
%  indidual TIFF after line- and motion-correction.
%  It also stores metadata in a separate .mat struct in the sourcedir.
%
% none :
%  This is NON-processed / raw data.  This is likely TEFO stuff for
%  now... more on this later.
%  Unprocessed TIFFs will contain SI's metadata in them, so metadata
%  can and should still be extracted.
%   
maskType = D.maskType;

switch maskType
    case 'circles'
        %refNum = D.maskInfo.refNum;
        maskPaths = D.maskInfo.maskPaths;
        slices = D.maskInfo.slices;
    case 'pixels'
        smoothXY = D.maskInfo.params.smoothXY;
        ksize = D.maskInfo.params.kernelXY;
        slices = D.maskInfo.slices;   
    case 'contours'
        %refNum = params.refNum;
        maskPaths = D.maskInfo.maskPaths;
        slices = D.maskInfo.slices;
end

tracesPath = fullfile(D.datastructPath, 'traces');
if ~exist(tracesPath, 'dir')
    mkdir(tracesPath);
end
acquisitionName = D.acquisitionName;
nTiffs = D.nTiffs;
nchannels = D.channelIdx;

meta = load(D.metaPath);

switch maskType
    case 'circles'
        
        for cidx=1:nchannels
            %T = struct();  
            
            for sidx = 1:length(slices)
                T = struct();  
                % Load manually-drawn circle ROIs:
                currSliceIdx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, currSliceIdx, length(slices));
                
                maskStruct=load(maskPaths{sidx});
                maskcell = maskStruct.maskcell; 
                % MASKCELL - this is a cell array where each ROI of the
                % current slice is stored as a sparse 
                
                % Load current slice movie and apply mask from refRun:
                for fidx = 1:nTiffs
                    %meta = load(metaPaths{fidx});                              % Load meta info for current file.
                    slicePath = meta.file(fidx).si.tiffPath;                                 % Get path to all slice TIFFs for current file.
                    
                    % TODO:  fix FIDX indexing so that cross-referenced MC
                    % tiffs correspond to to true File001, for ex., instead
                    % of File006 (i.e., relative to full collection of
                    % TIFFs, File001 of conditionRSVP might be File006 of a
                    % MC-analysis in which There were File001-File005 for
                    % conditionRetino or sth.):
                    currSliceName = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif',...
                                        acquisitionName, currSliceIdx, cidx, fidx);             % Use name of reference slice TIFF to get current slice fn
                    Y = tiffRead(fullfile(slicePath, currSliceName));               % Read in current file of current slice.
                    
                    % TO DO:  
                    % Check frame-to-frame correlation for bad
                    % motion correction:
                    checkframes = @(x,y) corrcoef(x,y);
                    refframe = 1;
                    corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,refframe)), 1:size(Y,3), 'UniformOutput', false);
                    corrs = cat(3, corrs{1:end});
                    meancorrs = squeeze(mean(mean(corrs,1),2));
                    badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*3); %find(meancorrs<0.795);
                    
                    if length(badframes)>1
                        fprintf('Bad frames found in movie %s at: %s\n', currSliceName, mat2str(badframes(2:end)));
                    end
                    while length(badframes) >= size(Y,3)*0.25
                        refframe = refframe +1;
                        corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,1)), 1:size(Y,3), 'UniformOutput', false);
                        corrs = cat(3, corrs{1:end});
                        meancorrs = squeeze(mean(mean(corrs,1),2));
                        badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*2); %find(meancorrs<0.795);
                    end
                        
                        
                    %find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*2)
                    
                    avgY = mean(Y, 3);

                    % Use masks to extract time-series trace of each ROI:
                    % --- TO DO --- FIX trace extraction to use sparse
                    % matrices from create_rois.m:
                    % ---------------
                    
%                     maskfunc = @(x,y) sum(sum(x.*y));
%                     tic()
%                     x = 1:size(Y,3);
%                     rawTraces2 = arrayfun(@(xval) cellfun(@(c) maskfunc(full(c), Y(:,:,xval)), maskcell, 'UniformOutput', false), x, 'UniformOutput', false)
%                     toc() % 100sec (looping over rois = 120sec)
%                     
                    tic()
                    %maskfunc = @(x,y) sum(sum(x.*y));
                    %cellY = num2cell(Y, [1 2]);
                    %rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(c) maskfunc(full(c), frame), masklog), cellY, 'UniformOutput', false));

                    maskfunc = @(x,y) sum(x(y)); % way faster
                    cellY = num2cell(Y, [1 2]);
                    % For each frame of the movie, apply each ROI mask:
                    rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(c) maskfunc(frame, c), maskcell), cellY, 'UniformOutput', false));
                    rawTraces = cat(1, rawTracesTmp{1:end});
                    
                    toc() % 44sec.
                    fprintf('Size rawTraces mat: %s\n', mat2str(size(rawTraces)));

                    % ---------------
                    
                    [nr,nc] = size(avgY);
                    nFrames = size(Y,3);
                    
%                     maskfunc = @(x,y) sum(sum(x.*y));
%                     cellY = num2cell(Y, [1 2]);
%                     %rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(roi) maskfunc(full(roi), frame), maskcell), cellY, 'UniformOutput', false));
%                     rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(roi) maskfunc(full(roi), frame), maskcell), cellY, 'UniformOutput', false));
% 
%                     rawTraces = cat(1, rawTracesTmp{1:end});
%                     fprintf('Size rawTraces mat: %s\n', mat2str(size(rawTraces)));
%                     
                    
%                     T.rawTraces.file{fidx} = rawTraces;
%                     %T.masks.file(fidx) = {masks};
%                     T.avgImage.file{fidx} = avgY;
%                     T.slicePath.file{fidx} = fullfile(slicePath, currSliceName);
%                     T.badFrames.file(fidx) = badframes;
%                     T.meancorrs.file{fidx} = meancorrs;
%                     T.refframe.file(fidx) = refframe;
                    
                    T.file(fidx).rawTraces = rawTraces;
                    %T.masks.file(fidx) = {masks};
                    T.file(fidx).avgImage = avgY;
                    T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
                    T.file(fidx).badFrames = badframes;
                    T.file(fidx).corrcoeffs = corrs;
                    T.file(fidx).refframe = refframe;
                    T.file(fidx).info.szFrame = size(avgY);
                    T.file(fidx).info.nFrames = nFrames;
                    T.file(fidx).info.nRois = length(maskcell);
                    
                    %clearvars rawTraces rawTracesTmp cellY Y avgY corrs
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);
                end

                % Save traces for each file to slice struct:                    
%                 tracesPath = fullfile(dstructPath, 'traces');
%                 if ~exist(tracesPath, 'dir')
%                     mkdir(tracesPath);
%                 end
                tracesName = sprintf('traces_Slice%02d_Channel%02d', currSliceIdx, cidx);
                fprintf('Saving struct: %s.\n', tracesName);
                
                save_struct(tracesPath, tracesName, T);
                %clearvars T maskStruct maskcell

            end
        end

    case 'pixels'
        % do stuff
        for cidx=1:nchannels

            for sidx=1:length(slices)
                T = struct();
                
                % Get slice names:
                currSliceIdx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, currSliceIdx, length(slices));
                
                for fidx=1:nTiffs %1:3:3 %nTiffs
                    %meta = load(metaPaths{fidx});
                    slicePath = meta.file(fidx).si.tiffPath;                                 % Get path to all slice TIFFs for current file.
                    currSliceName = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif',...
                                        acquisitionName, currSliceIdx, cidx, fidx);             % Use name of reference slice TIFF to get current slice fn
                    Y = tiffRead(fullfile(slicePath, currSliceName));               % Read in current file of current slice.
                    avgY = mean(Y, 3);
                    nFrames = size(Y,3);
                    % ------------
                    tic();
                    if smoothXY
                        fprintf('Smoothing with kernel size %i...\n', ksize);
                        for frame=1:size(Y,3);
                            currFrame = Y(:,:,frame);
                            padY = padarray(currFrame, [ksize, ksize], 'replicate');
                            convY = conv2(padY, fspecial('average',[ksize ksize]), 'same');
                            Y(:,:,frame) = convY(ksize+1:end-ksize, ksize+1:end-ksize);
                        end
                    end
                    toc();
                    % -----
                    
                    rawTraces = reshape(Y, [size(Y,1)*size(Y,2), size(Y,3)]);
                    
                    T.file(fidx).rawTraces = rawTraces'; % cols="rois" and rows=frames.
                    %T.masks.file(fidx) = {masks};
                    T.file(fidx).avgImage = avgY;
                    T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
                    %T.meta.file(fidx) = meta;
                    %T.file(fidx).refframe = refframe;
                    T.file(fidx).info.szFrame = size(avgY);
                    T.file(fidx).info.nFrames = nFrames;
                    T.file(fidx).info.nRois = size(rawTraces,1);
                    
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);
                end
                
                % Save traces for each file to slice struct:                    
%                 tracesPath = fullfile(dstructPath, 'traces');
%                 if ~exist(tracesPath, 'dir')
%                     mkdir(tracesPath);
%                 end

                tracesName = sprintf('traces_Slice%02d_Channel%02d', currSliceIdx, cidx);
                save_struct(tracesPath, tracesName, T);
            end   
        end
        
    case 'contours'
        % do other stuff
        for cidx=1:nchannels
           for sidx=1:length(slices)
                
                T = struct();  

                currSliceIdx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, currSliceIdx, length(slices));
                
                % Load masks for current slice:
                maskStruct=load(D.maskInfo.maskPaths{sidx});

                % Load current slice movie and apply mask from refRun:
                for fidx = 1:nTiffs
                    fidx
                    % Get current slice (corrected):
                    slicePath = meta.file(fidx).si.tiffPath;                                 % Get path to all slice TIFFs for current file.
                    currSliceName = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif',...
                                        acquisitionName, currSliceIdx, cidx, fidx);             % Use name of reference slice TIFF to get current slice fn
                    Y = tiffRead(fullfile(slicePath, currSliceName));               % Read in current file of current slice.
                    
                    % TODO:
                    % save maskcell in extraction step (getRoisNMf.m)
                    % instead...
                    % *********************************
                    % Get current masks:
                    %if ~isfield(maskStruct.file(fidx), 'maskmat') || isempty(maskStruct.file(fidx).maskmat)
                        roiMat = maskStruct.file(fidx).rois;
                        nRois = size(roiMat,2);
                        d1=maskStruct.file(fidx).options.d1;
                        d2=maskStruct.file(fidx).options.d2;
                        % To view ROIs:
                        %figure(); plot_contours(roiMat, imresize(mat2gray(avgY), [512 1024]), maskStruct.file(1).options);
                        
                        % Want a cell array if sparse matrices, each of which is a
                        % "mask" of an ROI.
                        % 1 ROI mask is:  full(reshape(Aor(:,i),d1,d2));
                        %maskcell = maskStruct.maskcell;
                        % MASKCELL - this is a cell array where each ROI of the
                        % current slice is stored as a sparse
                    %if ~isfield(maskStruct.file(fidx), 'maskcell') || isempty(maskStruct.file(fidx).maskcell)
                        tic()
                        maskcellTmp = arrayfun(@(roi) reshape(roiMat(:,roi), d1, d2), 1:nRois, 'UniformOutput', false);
                        if maskStruct.file(fidx).preprocessing.scaleFOV
                            maskcellTmp = arrayfun(@(roi) imresize(full(maskcellTmp{roi}), [size(maskcellTmp{roi},1)/2, size(maskcellTmp{roi},2)]), 1:nRois, 'UniformOutput', false);
                        end
                        maskcell = cellfun(@logical, maskcellTmp, 'UniformOutput', false); % Does this work on sparse mats?
                        maskcell = cellfun(@sparse, maskcell, 'UniformOutput', false);
                        maskStruct.file(fidx).maskcell = maskcell;
                        toc()

                        [fp,fn,fe] = fileparts(maskPaths{sidx});
                        save_struct(fp, strcat(fn,fe), maskStruct, 'append')
                    %else
                        %maskcell = maskStruct.file(fidx).maskcell;
%                             maskmat = arrayfun(@(roi) reshape(full(maskcell{roi}), [size(maskcell{roi},1)*size(maskcell{roi},2) 1]), 1:length(maskcell), 'UniformOutput', false);
%                             %maskmat = cat(2,maskmat{1:end});
%                             maskStruct.file(fidx).maskmat = maskmat;
%                             [fp,fn,fe] = fileparts(maskPaths{sidx});
%                             save_struct(fp, strcat(fn,fe), maskStruct, 'append')
                    %end
%                     else
%                         maskcell = maskStruct.file(fidx).maskcell;
%                         maskmat = maskStruct.file(fidx).maskmat;
%                     end

                    % Check frame-to-frame correlation for bad
                    % motion correction:
                    if ~maskStruct.file(fidx).preprocessing.removeBadFrame
                        checkframes = @(x,y) corrcoef(x,y);
                        refframe = 1;
                        corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,refframe)), 1:size(Y,3), 'UniformOutput', false);
                        corrs = cat(3, corrs{1:end});
                        meancorrs = squeeze(mean(mean(corrs,1),2));
                        badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*3); %find(meancorrs<0.795);

                        if length(badframes)>1
                            fprintf('Bad frames found in movie %s at: %s\n', currSliceName, mat2str(badframes(2:end)));
                        end
                        while length(badframes) >= size(Y,3)*0.25
                            refframe = refframe +1;
                            corrs = arrayfun(@(i) checkframes(Y(:,:,i), Y(:,:,1)), 1:size(Y,3), 'UniformOutput', false);
                            corrs = cat(3, corrs{1:end});
                            meancorrs = squeeze(mean(mean(corrs,1),2));
                            badframes = find(abs(meancorrs-mean(meancorrs))>=std(meancorrs)*2); %find(meancorrs<0.795);
                        end
                    else
                        refframe = maskStruct.file(fidx).preprocessing.refframe;
                        badframes = maskStruct.file(fidx).preprocessing.badframes;
                        corrs = maskStruct.file(fidx).preprocessing.corrcoeffs;
                    end
                                            
                    avgY = mean(Y, 3);
                    [nr,nc] = size(avgY);
                    nFrames = size(Y,3);

                    % Use masks to extract time-series trace of each ROI:
%                     tic()
%                     maskfunc = @(x,y) sum(sum(x.*y));
%                     %cellY = num2cell(Y, [1 2]);
%                     nFrames = size(Y,3);
%                     nRois = size(maskmat, 2);
%                     rawTracesTmp = arrayfun(@(frame) arrayfun(@(roi) maskfunc(reshape(maskmat(:,roi),nr,nc), Y(:,:,frame)), 1:nRois, 'UniformOutput', false), 1:nFrames, 'UniformOutput', false);
%                     rawTraces = cell2mat(cat(1, rawTracesTmp{1:end}));
%                     toc() % 44sec.

                    
%                     tic()
%                     maskfunc = @(x,y) sum(sum(x.*y));
%                     cellY = num2cell(Y, [1 2]);
%                     %rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(roi) maskfunc(full(roi), frame), maskcell), cellY, 'UniformOutput', false));
%                     rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(roi) maskfunc(full(roi), frame), maskcell), cellY, 'UniformOutput', false));
% 
%                     rawTraces = cat(2, rawTracesTmp{1:end});
%                     fprintf('Size rawTraces mat: %s\n', mat2str(size(rawTraces)));
%                     
%                     toc() % 20sec.
                    
                    tic()
                    % maskcell should be a cell array of logicals of size
                    % (1,nRois).  Each logical aray is a saprse mat whose
                    % full size is size(avgY).
                    maskfunc = @(x,y) sum(x(y)); % way faster
                    cellY = num2cell(Y, [1 2]);
                    % For each frame of the movie, apply each ROI mask:
                    rawTracesTmp = squeeze(cellfun(@(frame) cellfun(@(c) maskfunc(frame, c), maskcell), cellY, 'UniformOutput', false));
                    rawTraces = cat(1, rawTracesTmp{1:end});
                    toc() % 44sec.
                    fprintf('Size rawTraces mat: %s\n', mat2str(size(rawTraces)));
                    % ---------------   

                    T.file(fidx).rawTraces = rawTraces;
                    %T.masks.file(fidx) = {masks};
                    T.file(fidx).avgImage = avgY;
                    T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
                    T.file(fidx).badFrames = badframes;
                    T.file(fidx).corrcoeffs = corrs;
                    T.file(fidx).refframe = refframe;
                    T.file(fidx).info.szFrame = size(avgY);
                    T.file(fidx).info.nFrames = nFrames;
                    T.file(fidx).info.nRois = length(maskcell);
                    
                    
                    %clearvars rawTraces rawTracesTmp cellY Y avgY corrs
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);
                end

                % Save traces for each file to slice struct:                    
%                 tracesPath = fullfile(dstructPath, 'traces');
%                 if ~exist(tracesPath, 'dir')
%                     mkdir(tracesPath);
%                 end
                tracesName = sprintf('traces_Slice%02d_Channel%02d', currSliceIdx, cidx);
                fprintf('Saving struct: %s.\n', tracesName);
                
                save_struct(tracesPath, tracesName, T);
                %clearvars T maskStruct maskcell
            end
            
        end
        
    otherwise
        
        fprintf('Mask type %s not recognized...\n', maskType);
        fprintf('No traces extracted.\n')

end

tracesSaved = dir(fullfile(tracesPath, 'traces_*'));
tracesSaved = tracesSaved(arrayfun(@(x) ~strcmp(x.name(1),'.'),tracesSaved));
nSlicesTrace = length(tracesSaved);


end
