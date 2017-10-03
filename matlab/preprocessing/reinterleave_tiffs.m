function reinterleave_tiffs(A, source_dir, dest_dir,  split_channels)

    if ~exist('split_channels', 'var')
        split_channels = false;
    end
    if split_channels
        fprintf('Splitting channels bec big TIFF.\n');
    end

    nslices = length(A.slices);
    nchannels = A.nchannels;
    nfiles = A.ntiffs;
    nvolumes = A.nvolumes;
    nframes = nslices*nvolumes*nchannels;
    d1=A.lines_per_frame; d2=A.pixels_per_line;

    sliceidxs = 1:nchannels:nslices*nchannels;
    tiffnames= dir(fullfile(source_dir, '*.tif'));
    tiffnames = {tiffnames(:).name}';
    fprintf('Found %i tiffs total for re-interleaving.\n', length(tiffnames));
    for fi=1:nfiles
         newtiff = zeros(d1, d2, nframes);
         for sl = 1:nslices
             for ch=1:nchannels
                 suffix = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', A.acquisition, sl, ch, fi)
                 curr_tiff_idx = find(arrayfun(@(fn) length(strfind(tiffnames{fn}, suffix)), 1:length(tiffnames)))
                 
                 % This currently assumes only 2 channels...
                 currtiff = tiffnames{curr_tiff_idx};
                 [tmp,~] = tiffRead(fullfile(source_dir,currtiff));
                 if ch==1
                     newtiff(:,:,sliceidxs(sl):(nslices*nchannels):end) = tmp;
                 else
                     newtiff(:,:,(sliceidxs(sl)+1):(nslices*nchannels):end) = tmp;
                 end
             end
         end
         if split_channels
             for ch=1:nchannels
                newtiffname = sprintf('%s_File%03d_Channel%02d.tif', A.acquisition, fi, ch);
                tiffWrite(newtiff(:,:,ch:nchannels:end), newtiffname, dest_dir, 'int16');
             end 
         else
             newtiffname = sprintf('%s_File%03d.tif', A.acquisition, fi);
             tiffWrite(newtiff, newtiffname, dest_dir, 'int16');
         end
     end

end
