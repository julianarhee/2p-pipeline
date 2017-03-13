function get_traces(dstructPath, mask_type, acquisition_name, nTiffs, nchannels, metaPaths, varargin)

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

nvargin = length(varargin);
switch nvargin
    case 1
        maskInfo = varargin{1};
        refNum = maskInfo.refNum;
        maskPaths = maskInfo.maskPaths;
        slices = maskInfo.slicesToUse;
    case 2
        params = varargin{2};
        smooth_xy = params.smoothXY;
        ksize = params.kernelXY;
        slices = params.slicesToUse;   
end


switch mask_type
    case 'circles'
        
        for cidx=1:nchannels
            %T = struct();  
            
            parfor sidx = 1:length(slices)
                T = struct();  
                % Load manually-drawn circle ROIs:
                currSliceIdx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, currSliceIdx, length(slices));
                
                M=load(maskPaths{sidx});
                masks = M.masks;
                [channelPath, refDir, ~] = fileparts(M.refPath);
                
                % Load current slice movie and apply mask from refRun:
                for fidx = 1:nTiffs
                    metaFile = load(metaPaths{fidx});                              % Load meta info for current file.
                    slicePath = metaFile.tiffPath;                                 % Get path to all slice TIFFs for current file.
                    currSliceName = strrep(M.slice, sprintf('File%03d', refNum),...
                                            sprintf('File%03d', fidx));             % Use name of reference slice TIFF to get current slice fn
                    Y = tiffRead(fullfile(slicePath, currSliceName));               % Read in current file of current slice.
                    avgY = mean(Y, 3);

                    % Use masks to extract time-series trace of each ROI:
                    rawTraces = zeros(size(masks,3), size(Y,3));
                    for r=1:size(masks,3)
                        currMask = masks(:,:,r);
                        maskedY = nan(1,size(Y,3));
                        for t=1:size(Y,3)
                            tmpMasked = currMask.*Y(:,:,t);
                            %tmpMasked(tmpMasked==0) = NaN;
                            %maskedY(t) = nanmean(tmpMasked(:));
                            maskedY(t) = sum(tmpMasked(:));
                        end
                        rawTraces(r,:,:) = maskedY;
                    end

                    T.traces.file(fidx) = {rawTraces};
                    %T.masks.file(fidx) = {masks};
                    T.avgImage.file(fidx) = {avgY};
                    T.slicePath.file{fidx} = fullfile(slicePath, currSliceName);
                    %T.meta.file(fidx) = metaFile;
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);
                end

                % Save traces for each file to slice struct:                    
                tracesPath = fullfile(dstructPath, 'traces');
                if ~exist(tracesPath, 'dir')
                    mkdir(tracesPath);
                end
                tracesName = sprintf('traces_Slice%02d_Channel%02d', currSliceIdx, cidx);
                fprintf('Saving struct: %s.\n', tracesName);
                
                save_struct(tracesPath, tracesName, T);
                %clear T masks M

            end
        end

    case 'pixels'
        % do stuff
        for channel=1:nchannels

            parfor sidx=1:length(slices)
                T = struct();
                
                % Get slice names:
                currSliceIdx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, currSliceIdx, length(slices));

                for fidx=1:nTiffs %1:3:3 %nTiffs
                    metaFile = load(metaPaths{fidx});
                    slicePath = metaFile.tiffPath;
                    currSliceName = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, currSliceIdx, cidx, fidx);
                    
                    Y = tiffRead(fullfile(slicePath, currSliceName));
                    avgY = mean(Y, 3);

                    % ------------
                    if smooth_spatial==1
                        fprintf('Smoothing with kernel size %i...\n', ksize);
                        for frame=1:size(Y,3);
                            currFrame = Y(:,:,frame);
                            padY = padarray(currFrame, [ksize, ksize], 'replicate');
                            convY = conv2(padY, fspecial('average',[ksize ksize]), 'same');
                            Y(:,:,frame) = convY(ksize+1:end-ksize, ksize+1:end-ksize);
                        end
                    end
                    
                    rawTraces = reshape(Y, [size(Y,1)*size(Y,2), size(Y,3)]);
                    
                    T.traces.file(fidx) = {rawTraces};
                    T.avgImage.file(fidx) = {avgY};
                    T.slicePath.file(fidx) = fullfile(slicePath, currSliceName);
                    T.meta.file(fidx) = metaFile;
                    T.params.file(fidx) = params;
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);
                end
                
                % Save traces for each file to slice struct:                    
                tracesPath = fullfile(dstructPath, 'traces');
                if ~exist(tracesPath, 'dir')
                    mkdir(tracesPath);
                end

                tracesName = sprintf('traces_Slice%02d_Channel%02d.tif', sidx, cidx);
                save_struct(tracesPath, tracesName, T);
            end   
        end
        
    case 'nmf'
        % do other stuff
        parfor channel=1:nchannels
            T = struct();

            for sidx=1:length(slices)

               % Get slice names:
               currSliceIdx = slices(sidx);
               for fidx=1:nTiffs
                   metaFile = load(metaPaths{fidx});
                   slicePath = metaFile.tiffPath;
                   currSliceName = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, currSliceIdx, cidx, fidx);

                   Y = tiffRead(fullfile(slicePath, currSliceName));
                   avgY = mean(Y, 3);

                   % DO STUFF
                   %
                   %
                    %

               end
              
               % Save traces for each file to slice struct:                    
               tracesPath = fullfile(dstructPath, 'traces');
               if ~exist(tracesPath, 'dir')
                   mkdir(tracesPath);
               end

               tracesName = sprintf('traces_Slice%02d_Channel%02d.tif', sidx, cidx);
               save_struct(tracesPath, tracesName, T);
                
             end
        end
        
    otherwise
        
        fprintf('Mask type %s not recognized...\n', mask_type);
        fprintf('No traces extracted.\n')

end

end


%             T = struct();
%             for curr_slice = 10:2:12 %1:tiff_info.nslices %12:2:16 %1:tiff_info.nslices
%                 slice_indices = curr_slice:tiff_info.nframes_per_volume:tiff_info.ntotal_frames;
%                 
%                 %sample_tiff = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice, channel_idx, 1);
%                 %sampleY = tiffRead(fullfile(tiff_info.tiffPath, sample_tiff));
%                 %avg_sampleY = mean(sampleY, 3);
%                 %masks = ROIselect_circle(mat2gray(avg_sampleY));
%                 
%                 for channel_idx=1:nchannels
%                     
%                     for curr_file=1:nTiffs %1:3:3 %nTiffs
%                         curr_tiff = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice, channel_idx, curr_file);
%                         Y = tiffRead(fullfile(tiff_info.tiffPath, curr_tiff));
%                         avgY = mean(Y, 3);
%                         
%                         % ------------
%                         if smooth_spatial==1
%                             fprintf('Smoothing with kernel size %i...\n', ksize);
%                             for frame=1:size(Y,3);
%                                 currFrame = Y(:,:,frame);
%                                 padY = padarray(currFrame, [ksize, ksize], 'replicate');
%                                 convY = conv2(padY, fspecial('average',[ksize ksize]), 'same');
%                                 Y(:,:,frame) = convY(ksize+1:end-ksize, ksize+1:end-ksize);
%                             end
%                         end
%                         % -------------
%                         switch roi_type
%                             case 'create_rois'
%                                 masks = ROIselect_circle(mat2gray(avgY));
%                                 % TODO:  add option for different type of
%                                 % manual ROI creation?  
%                                 rawTraces = zeros(size(masks,3), size(Y,3));
%                                 for r=1:size(masks,3)
%                                     currMask = masks(:,:,r);
%                                     maskedY = nan(1,size(Y,3));
%                                     for t=1:size(Y,3)
%                                         tmpMasked = currMask.*Y(:,:,t);
%                                         tmpMasked(tmpMasked==0) = NaN;
%                                         maskedY(t) = nanmean(tmpMasked(:));
%                                         %maskedY(t) = sum(tmpMasked(:));
%                                     end
%                                     rawTraces(r,:,:) = maskedY;
%                                 end
%                                 paramspecs = '';
%                                 
%                             case 'pixels'
%                                 masks = 'pixels';
%                                 rawTraces = Y;
%                                 paramspecs = sprintf('K%i', ksize);
%                                 
%                             otherwise
%                                 % TODO:  fix this so can decide which
%                                 % slice's masks to use (and which run
%                                 % no's)? -- otherwise have to make masks
%                                 % for every slice for every file.... 
%                                 
%                                 % Use previously-defined masks: 
%                                 [prev_fn, prevPath, ~] = uigetfile();
%                                 prev_struct = load(fullfile(prevPath, prev_fn));
%                                 %prev_struct_fieldname = fieldnames(prev_struct);
%                                 masks = prev_struct.slice(curr_slice).masks;
%                         end
%                         % ------------
% %                         T.slice(curr_slice).avgImage.file(curr_file) = {avgY};
% %                         T.slice(curr_slice).traces.file(curr_file) = {rawTraces};
% %                         T.slice(curr_slice).masks.file(curr_file) = {masks};
% %                         T.slice(curr_slice).frame_indices = slice_indices;
%                         T.avgImage.file(curr_file) = {avgY};
%                         T.traces.file(curr_file) = {rawTraces};
%                         T.masks.file(curr_file) = {masks};
%                         T.frame_indices = slice_indices;
%                     end
%                     [pathstr,name,ext] = fileparts(tiff_info.tiffPath);
%                     struct_savePath = fullfile(pathstr, 'datastructs', 'traces');
%                     if ~exist(struct_savePath, 'dir')
%                         mkdir(struct_savePath);
%                     end
%                     
%                     curr_tracestruct_name = char(sprintf('traces_Slice%02d_nFiles%i_%s%s.mat', curr_slice, nTiffs, roi_type, paramspecs));
%                     save(fullfile(struct_savePath, curr_tracestruct_name), 'T', '-v7.3');
%                    
%                     if length(trace_struct_names)<1
%                         trace_struct_names{1} = curr_tracestruct_name;
%                     else
%                         trace_struct_names{end+1} = curr_tracestruct_name;
%                     end
%                 end
%                                 
%             end
%             
%         case 'raw'
%             % do other stuff
%             
%             
%         otherwise
%             fprintf('No TIFFs found...\n')
%     end
%     
%     trace_info.struct_fns = trace_struct_names;
%     trace_info.corrected = corrected;
%     trace_info.acquisition_name = acquisition_name;
%     trace_info.nTiffs = nTiffs;
%     trace_info.nchannels = nchannels;
%     trace_info.roi_type = roi_type;
%     trace_info.paramspec = paramspecs;
%     trace_info.smoothed_xy = smooth_spatial;
%     trace_info.kernel_xy = ksize;
%     trace_info_fn = char(sprintf('info_nFiles%i_%s%s.mat', nTiffs, roi_type, paramspecs));
%     save(fullfile(struct_savePath, trace_info_fn));
%     
% end
