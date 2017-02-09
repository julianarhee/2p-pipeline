function T = get_raw_traces(corrected, acquisition_name, ntiffs, nchannels, tiff_info, roi_type, channel_idx)
    
    switch corrected
        case 'acquisition2p'
            T = struct();
            for curr_slice = 10:15 %1:tiff_info.nslices
                slice_indices = curr_slice:tiff_info.nframes_per_volume:tiff_info.ntotal_frames;
                
                sample_tiff = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice, channel_idx, 1);
                sampleY = tiffRead(fullfile(tiff_info.tiff_path, sample_tiff));
                avg_sampleY = mean(sampleY, 3);
                masks = ROIselect_circle(mat2gray(avg_sampleY));
                
                for channel_idx=1:nchannels
                    
                    for curr_file=1:ntiffs
                        curr_tiff = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, curr_slice, channel_idx, curr_file);
                        Y = tiffRead(fullfile(tiff_info.tiff_path, curr_tiff));
%                         
%                         %varargout{1} = img;
%                         % rescale if needed?
% %                         [iy, ix, iz] = size(Y);
% %                         ny=iy*yscale;nx=ix*xscale;nz=iz; %% desired output dimensions
% %                         [y x z]=...
% %                            ndgrid(linspace(1,size(Y,1),ny),...
% %                                   linspace(1,size(Y,2),nx),...
% %                                   linspace(1,size(Y,3),nz));
% %                         imOut=interp3(Y,x,y,z);
% %                         varargout{1} = imOut;
%                         scale_vec = [2 1 1];
%                         T = maketform('affine',[scale_vec(1) 0 0; 0 scale_vec(2) 0; 0 0 scale_vec(3); 0 0 0;]);
%                         R = makeresampler({'cubic','cubic','cubic'},'fill');
%                         ImageScaled = tformarray(Y,T,R,[1 2 3],[1 2 3], size(Y).*scale_vec,[],0);


                        avgY = mean(Y, 3);
                        
                        % ---
                        switch roi_type
                            case 'create_rois'
                                %masks = ROIselect_circle(mat2gray(avgY));
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
                            case 'smoothed_pixels'
                                raw_traces = zeros(size(Y));
                                for frame=1:size(Y,3);
                                    curr_frame = Y(:,:,frame);
                                    padY = padarray(curr_frame, [ksize, ksize], 'replicate');
                                    convY = conv2(padY, fspecial('average',[ksize ksize]), 'same');
                                    raw_traces(:,:,frame) = convY(ksize+1:end-ksize, ksize+1:end-ksize);
                                end
                            otherwise
                                % TODO:  fix this so can decide which
                                % slice's masks to use (and which run
                                % no's)? -- otherwise have to make masks
                                % for every slice for every file.... 
                                
                                % Use previously-defined masks: 
                                [prev_fn, prev_path, ~] = uigetfile();
                                prev_struct = load(fullfile(prev_path, prev_fn));
                                %prev_struct_fieldname = fieldnames(prev_struct);
                                masks = prev_struct.slice(curr_slice).masks;
                        end
        


                        % ---
                       
                        T.slice(curr_slice).traces.file{curr_file} = raw_traces;
                        %T.slice(curr_slice).masks.file{curr_file} = raw_traces;
                        T.slice(curr_slice).masks = masks;
                        T.slice(curr_slice).frame_indices = slice_indices;
                    end
                end
                
            end
            
        case 'none'
            % do other stuff
        otherwise
            fprintf('No TIFFs found...\n')
    end


end
