function create_averaged_slices(deinterleaved_tiff_basepath, average_slices_basepath, I, A)

nfiles = A.ntiffs;
nchannels = A.nchannels;
simeta = load(A.raw_simeta_path);

if I.corrected
    allmcparams = load(A.mcparams_path);
    mcparams = allmcparams.(I.mc_id);
    clear allmcparams
end

if ~exist(average_slices_basepath, 'dir')
    mkdir(average_slices_basepath)
end

fprintf('Creating average slices and saving to:\n%s\n', average_slices_basepath);

for tiff_idx=1:nfiles %length(data_files)
      
    for ch=1:nchannels
        ch_path = fullfile(average_slices_basepath, sprintf('Channel%02d', ch), sprintf('File%03d', tiff_idx));
        ch_path_vis = fullfile(average_slices_basepath, sprintf('Channel%02d', ch), sprintf('File%03d_visible', tiff_idx));
        if ~exist(ch_path, 'dir')
            mkdir(ch_path)
            mkdir(ch_path_vis)
        end 
        fprintf('Averaging Channel %i, File %i...\n', ch, tiff_idx);

        slice_dir = fullfile(deinterleaved_tiff_basepath, sprintf('Channel%02d', ch), sprintf('File%03d', tiff_idx));

        tiffs = dir(fullfile(slice_dir, '*.tif'));
        tiffs = {tiffs(:).name}';
        d3 = length(tiffs);  % This assumes that parsed files have been sorted into standard Channel-File-Slice format.
        %sample = read_file(fullfile(slice_dir, tiffs{1}));
        currtiffpath = fullfile(slice_dir, tiffs{1});
        curr_file_name = sprintf('File%03d', tiff_idx);
        if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
            sample = read_file(currtiffpath);
        else
            sample = read_imgdata(currtiffpath);
        end
        d1=size(sample,1); d2=size(sample,2); clear sample
	
        avgs = zeros([d1,d2,d3]);
        for sl=1:d3
            %tiffdata = read_file(fullfile(slice_dir, tiffs{sl}));
            currtiffpath = fullfile(slice_dir, tiffs{sl});
            curr_file_name = sprintf('File%03d', tiff_idx);
            if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
                tiffdata = read_file(currtiffpath);
            else
                tiffdata = read_imgdata(currtiffpath);
            end
    
            fprintf('TIFF %i (slice %i) of %i: size is %s.\n', tiff_idx, sl, length(tiffs), mat2str(size(tiffdata)));
            avgs(:,:,sl) = mean(tiffdata, 3);
            slicename = sprintf('average_Slice%02d_Channel%02d_File%03d.tif', sl, ch, tiff_idx);
            tiffWrite(avgs(:,:,sl), slicename, ch_path, 'int16');	
            fprintf('Saved slice %s to path %s.\n', slicename, ch_path)
        end
        
        % make visible
        tmp = (avgs-min(avgs(:)))./(max(avgs(:))-min(avgs(:)));
        for sl=1:d3       
            avgs_visible = adapthisteq(tmp(:,:,sl));
            slicename_vis = sprintf('average_Slice%02d_Channel%02d_File%03d_vis.tif', sl, ch, tiff_idx);
            tiffWrite(avgs_visible*((2^16)-1), slicename_vis, ch_path_vis); % default dtype=uint16
        end
    end
end

end
