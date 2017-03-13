function trace_struct = get_traces(datastruct_path, mask_type, acquisition_name, ntiffs, nchannels, meta_paths, varargin)

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
        slices = maskInfo.slices_to_use;
    case 2
        params = varargin{2};
        smooth_xy = params.smooth_xy;
        ksize = params.kernel_xy;
        slices = params.slices_to_use;   
end


switch mask_type
    case 'circles'
        
        for cidx=1:nchannels
            %T = struct();  
            
            parfor sidx = 1:length(slices)
                T = struct();  
                % Load manually-drawn circle ROIs:
                curr_slice_idx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, curr_slice_idx, length(slices));
                
                M=load(maskPaths{sidx});
                M = M.M;
                masks = M.masks;
                [channelPath, refDir, ~] = fileparts(M.refPath);
                refSlice = M.slice;

                % Load current slice movie and apply mask from refRun:
                for fidx = 1:ntiffs
                    metaFile = load(meta_paths{fidx});                              % Load meta info for current file.
                    slicePath = metaFile.tiff_path;                                 % Get path to all slice TIFFs for current file.
                    curr_slice_fn = strrep(M.slice, sprintf('File%03d', refNum),...
                                            sprintf('File%03d', fidx));             % Use name of reference slice TIFF to get current slice fn
                    Y = tiffRead(fullfile(slicePath, curr_slice_fn));               % Read in current file of current slice.
                    avgY = mean(Y, 3);

                    % Use masks to extract time-series trace of each ROI:
                    raw_traces = zeros(size(masks,3), size(Y,3));
                    for r=1:size(masks,3)
                        curr_mask = masks(:,:,r);
                        Y_masked = nan(1,size(Y,3));
                        for t=1:size(Y,3)
                            t_masked = curr_mask.*Y(:,:,t);
                            t_masked(t_masked==0) = NaN;
                            Y_masked(t) = nanmean(t_masked(:));
                            %Y_masked(t) = sum(t_masked(:));
                        end
                        raw_traces(r,:,:) = Y_masked;
                    end

                    T.traces.file(fidx) = {raw_traces};
                    T.masks.file(fidx) = {masks};
                    T.avg_image.file(fidx) = {avgY};
                    T.slice_path.file{fidx} = fullfile(slicePath, curr_slice_fn);
                    T.meta.file(fidx) = metaFile;
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, ntiffs);
                end

                % Save traces for each file to slice struct:                    
                traces_path = fullfile(datastruct_path, 'traces');
                if ~exist(traces_path, 'dir')
                    mkdir(traces_path);
                end
                traces_fn = sprintf('traces_Slice%02d_Channel%02d', curr_slice_idx, cidx);
                save_struct(traces_path, traces_fn, T);

            end
        end

    case 'pixels'
        % do stuff
        for channel=1:nchannels

            parfor sidx=1:length(slices)
                T = struct();
                
                % Get slice names:
                curr_slice_idx = slices(sidx);
                fprintf('Processing %i (slice %i) of %i SLICES.\n', sidx, curr_slice_idx, length(slices));

                for fidx=1:ntiffs %1:3:3 %ntiffs
                    metaFile = load(meta_paths{fidx});
                    slicePath = metaFile.tiff_path;
                    curr_slice_fn = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice_idx, cidx, fidx);
                    
                    Y = tiffRead(fullfile(slicePath, curr_slice_fn));
                    avgY = mean(Y, 3);

                    % ------------
                    if smooth_spatial==1
                        fprintf('Smoothing with kernel size %i...\n', ksize);
                        for frame=1:size(Y,3);
                            curr_frame = Y(:,:,frame);
                            padY = padarray(curr_frame, [ksize, ksize], 'replicate');
                            convY = conv2(padY, fspecial('average',[ksize ksize]), 'same');
                            Y(:,:,frame) = convY(ksize+1:end-ksize, ksize+1:end-ksize);
                        end
                    end
                    
                    raw_traces = reshape(Y, [size(Y,1)*size(Y,2), size(Y,3)]);
                    
                    T.traces.file(fidx) = {raw_traces};
                    T.avg_image.file(fidx) = {avgY};
                    T.slice_path.file(fidx) = fullfile(slicePath, curr_slice_fn);
                    T.meta.file(fidx) = metaFile;
                    T.params.file(fidx) = params;
                    
                    fprintf('Extracted traces for %i of %i FILES.\n', fidx, ntiffs);
                end
                
                % Save traces for each file to slice struct:                    
                traces_path = fullfile(datastruct_path, 'traces');
                if ~exist(traces_path, 'dir')
                    mkdir(traces_path);
                end

                traces_fn = sprintf('traces_Slice%02d_Channel%02d.tif', sidx, cidx);
                save_struct(fullfile(traces_path, traces_fn), 'T', '-v7.3');
            end   
        end
        
    case 'nmf'
        % do other stuff
        parfor channel=1:nchannels
            T = struct();

            for sidx=1:length(slices)

               % Get slice names:
               curr_slice_idx = slices(sidx);
               for fidx=1:ntiffs
                   metaFile = load(meta_paths{fidx});
                   slicePath = metaFile.tiff_path;
                   curr_slice_fn = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice_idx, cidx, fidx);

                   Y = tiffRead(fullfile(slicePath, curr_slice_fn));
                   avgY = mean(Y, 3);

                   % DO STUFF
                   %
                   %
                    %

               end
              
               % Save traces for each file to slice struct:                    
               traces_path = fullfile(datastruct_path, 'traces');
               if ~exist(traces_path, 'dir')
                   mkdir(traces_path);
               end

               traces_fn = sprintf('traces_Slice%02d_Channel%02d.tif', sidx, cidx);
               save(fullfile(traces_path, traces_fn), 'T', '-v7.3');
                
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
%                 %sampleY = tiffRead(fullfile(tiff_info.tiff_path, sample_tiff));
%                 %avg_sampleY = mean(sampleY, 3);
%                 %masks = ROIselect_circle(mat2gray(avg_sampleY));
%                 
%                 for channel_idx=1:nchannels
%                     
%                     for curr_file=1:ntiffs %1:3:3 %ntiffs
%                         curr_tiff = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice, channel_idx, curr_file);
%                         Y = tiffRead(fullfile(tiff_info.tiff_path, curr_tiff));
%                         avgY = mean(Y, 3);
%                         
%                         % ------------
%                         if smooth_spatial==1
%                             fprintf('Smoothing with kernel size %i...\n', ksize);
%                             for frame=1:size(Y,3);
%                                 curr_frame = Y(:,:,frame);
%                                 padY = padarray(curr_frame, [ksize, ksize], 'replicate');
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
%                                 raw_traces = zeros(size(masks,3), size(Y,3));
%                                 for r=1:size(masks,3)
%                                     curr_mask = masks(:,:,r);
%                                     Y_masked = nan(1,size(Y,3));
%                                     for t=1:size(Y,3)
%                                         t_masked = curr_mask.*Y(:,:,t);
%                                         t_masked(t_masked==0) = NaN;
%                                         Y_masked(t) = nanmean(t_masked(:));
%                                         %Y_masked(t) = sum(t_masked(:));
%                                     end
%                                     raw_traces(r,:,:) = Y_masked;
%                                 end
%                                 paramspecs = '';
%                                 
%                             case 'pixels'
%                                 masks = 'pixels';
%                                 raw_traces = Y;
%                                 paramspecs = sprintf('K%i', ksize);
%                                 
%                             otherwise
%                                 % TODO:  fix this so can decide which
%                                 % slice's masks to use (and which run
%                                 % no's)? -- otherwise have to make masks
%                                 % for every slice for every file.... 
%                                 
%                                 % Use previously-defined masks: 
%                                 [prev_fn, prev_path, ~] = uigetfile();
%                                 prev_struct = load(fullfile(prev_path, prev_fn));
%                                 %prev_struct_fieldname = fieldnames(prev_struct);
%                                 masks = prev_struct.slice(curr_slice).masks;
%                         end
%                         % ------------
% %                         T.slice(curr_slice).avg_image.file(curr_file) = {avgY};
% %                         T.slice(curr_slice).traces.file(curr_file) = {raw_traces};
% %                         T.slice(curr_slice).masks.file(curr_file) = {masks};
% %                         T.slice(curr_slice).frame_indices = slice_indices;
%                         T.avg_image.file(curr_file) = {avgY};
%                         T.traces.file(curr_file) = {raw_traces};
%                         T.masks.file(curr_file) = {masks};
%                         T.frame_indices = slice_indices;
%                     end
%                     [pathstr,name,ext] = fileparts(tiff_info.tiff_path);
%                     struct_save_path = fullfile(pathstr, 'datastructs', 'traces');
%                     if ~exist(struct_save_path, 'dir')
%                         mkdir(struct_save_path);
%                     end
%                     
%                     curr_tracestruct_name = char(sprintf('traces_Slice%02d_nFiles%i_%s%s.mat', curr_slice, ntiffs, roi_type, paramspecs));
%                     save(fullfile(struct_save_path, curr_tracestruct_name), 'T', '-v7.3');
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
%     trace_info.ntiffs = ntiffs;
%     trace_info.nchannels = nchannels;
%     trace_info.roi_type = roi_type;
%     trace_info.paramspec = paramspecs;
%     trace_info.smoothed_xy = smooth_spatial;
%     trace_info.kernel_xy = ksize;
%     trace_info_fn = char(sprintf('info_nFiles%i_%s%s.mat', ntiffs, roi_type, paramspecs));
%     save(fullfile(struct_save_path, trace_info_fn));
%     
% end
