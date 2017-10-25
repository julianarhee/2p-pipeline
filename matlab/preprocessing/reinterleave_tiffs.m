function reinterleave_tiffs(A, source_dir, dest_dir,  split_channels)

    simeta = load(A.raw_simeta_path);

    if ~exist('split_channels', 'var')
        split_channels = false;
    end
    if split_channels
        fprintf('Splitting channels bec big TIFF.\n');
    end

    if ~exist(dest_dir, 'dir')
        mkdir(dest_dir);
    end
    fprintf('Writing interleaved TIFFs to: %s\n', dest_dir);

    nslices = length(A.slices);
    nchannels = A.nchannels;
    nfiles = A.ntiffs;
    nvolumes = A.nvolumes;
    nframes = nslices*nvolumes*nchannels;
    d1=A.lines_per_frame; d2=A.pixels_per_line;

    sliceidxs = 1:nchannels:nslices*nchannels;
    tiffnames= dir(fullfile(source_dir, '*.tif'));
    tiffnames = {tiffnames(:).name}';
    fprintf('Found %i tiffs total for interleaving.\n', length(tiffnames));
    if length(tiffnames)>0
        for fi=1:nfiles
             fprintf('d1: %i, d2: %i\n', d1, d2);
%              if length(A.slices)==1 && A.nchannels==1
%                  currfile = sprintf('File%03d', fi);
%                  channelnum = simeta.(currfile).SI.hChannels.channelSave;
%                  suffix = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', A.base_filename, 1, channelnum, fi);
%                  curr_tiff_idx = find(arrayfun(@(fn) length(strfind(tiffnames{fn}, suffix)), 1:length(tiffnames)));
 
%                  currtiff = tiffnames{fi};
%                  currtiffpath = fullfile(source_dir, currtiff);
%                  curr_file_name = sprintf('File%03d', fi);
% 
%                  % Don't really need to "interleave"
%                  newtiffname = sprintf('%s_File%03d.tif', A.base_filename, fi);
%                  copyfile(currtiffpath, fullfile(dest_dir, newtiffname));
%                fprintf('No need to reinterleave. Single channel, single slice.\n');
             newtiff = zeros(d1, d2, nframes);
             for sl = 1:nslices
                 for ch=1:nchannels
                     suffix = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', A.base_filename, sl, ch, fi);
                     curr_tiff_idx = find(arrayfun(@(fn) length(strfind(tiffnames{fn}, suffix)), 1:length(tiffnames)));
                     
                     % This currently assumes only 2 channels...
                     currtiff = tiffnames{curr_tiff_idx};
                     currtiffpath = fullfile(source_dir, currtiff);
                     curr_file_name = sprintf('File%03d', fi);
                     if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
                         [tmp,~] = tiffRead(fullfile(source_dir,currtiff));
                     else
                         tmp = read_imgdata(currtiffpath);
                     end
                     if ch==1
                         newtiff(:,:,sliceidxs(sl):(nslices*nchannels):end) = tmp;
                     else
                         newtiff(:,:,(sliceidxs(sl)+1):(nslices*nchannels):end) = tmp;
                     end
                 end
             end
             if split_channels
                 for ch=1:nchannels
                    newtiffname = sprintf('%s_File%03d_Channel%02d.tif', A.base_filename, fi, ch);
                    tiffWrite(newtiff(:,:,ch:nchannels:end), newtiffname, dest_dir, 'int16');
                 end 
             else
                 newtiffname = sprintf('%s_File%03d.tif', A.base_filename, fi);
                 tiffWrite(newtiff, newtiffname, dest_dir, 'int16');
             end
%              end
             fprintf('Done interleaving %s of %i files.\n', curr_file_name, length(tiffnames));
        end
    end
    if split_channels
         fprintf('Finished interleaving TIFFs. There should be %i TIFFs in specified destination dir.', nfiles*nchannels);
    else
        fprintf('Finished interleaving TIFFs. There should be %i TIFFs in specified destination dir.', nfiles)
    end

end
