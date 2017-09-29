function reinterleave_tiffs(A, tiffdir, movsize, split_channels)
    if ~exist('split_channels', 'var')
        split_channels = false;
    end
    if split_channels
        fprintf('Splitting channels bec big TIFF.\n');
    end
%     nslices = length(obj.correctedMovies.slice);
%     nchannels = length(obj.correctedMovies.slice(1).channel);
%     nfiles = length(obj.correctedMovies.slice(1).channel(1).fileName);
% 
%     movsize = obj.correctedMovies.slice(1).channel(1).size(1,:);
%     nframes = nslices*movsize(3)*nchannels;
    nslices = length(A.slices);
    nchannels = A.nchannels;
    nfiles = A.ntiffs;
    nvolumes = movsize(3);
    nframes = nslices*nvolumes*nchannels;

    sliceidxs = 1:nchannels:nslices*nchannels;
    tiffnames= dir(fullfile(tiffdir, '*.tif'));
    fprintf('Found %i tiffs total for re-interleaving.\n', length(tiffnames));

    for fi=1:nfiles
        newtiff = zeros(movsize(1), movsize(2), nframes);
        curr_file = sprintf('File%02d', fi);
        for sl = 1:nslices
            for ch=1:nchannels
                suffix = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition_name, sl, ch, fi); 
                curr_tiff_idx = find(arrayfun(@(fn) length(strfind(tiffnames{fn}, suffix)), 1:length(tiffnames)))
                
                % This currently assumes only 2 channels...
                % currtiff = obj.correctedMovies.slice(slice).channel(1).fileName{file};
                currtiff = tiffnames{curr_tiff_idx};
                [tmp,~] = tiffRead(fullfile(tiffdir,currtiff));
                if ch==1
                    newtiff(:,:,sliceidxs(slice):(nslices*nchannels):end) = tmp;
                else
                %currtiff = obj.correctedMovies.slice(slice).channel(2).fileName{file};
                %[tmp,~] = tiffRead(currtiff); 
                    newtiff(:,:,(sliceidxs(slice)+1):(nslices*nchannels):end) = tmp;
                end
            end
        end
        [fpath, fname, fext] = fileparts(fullfile(tiffdir, currtiff)); %obj.correctedMovies.slice(slice).channel(1).fileName{file});
        filename_parts = strsplit(fname, '_');
        if split_channels
            for cidx=1:nchannels
               newtiffname = strcat(strjoin(filename_parts(1:end-3), '_'), sprintf('_File%03d', file), sprintf('_Channel%02d', cidx), fext)
               tiffWrite(newtiff(:,:,cidx:nchannels:end), newtiffname, A.data_dir);
            end 
        else
            newtiffname = strcat(strjoin(filename_parts(1:end-3), '_'), sprintf('_File%03d', file), fext)
            tiffWrite(newtiff, newtiffname, A.data_dir); %, 'int16');
        end
    end
    

end
