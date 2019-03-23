function do_bidi_correction(paramspath, refpath)

% Reads in TIFFs and does correction for bidirection-scanning.
% For each TIFF in acquisition, does correction on interleaved tiff, saves to dir:  <source>_Bidi
% Also saves deinterleaved tiffs to dir:  <source>/bidi_deinterleaved dir

% INPUTs:
% source - can be 'Corrected' or 'Parsed' (i.e., do correcitonon mc or raw data)
% varargin - if no interleaved TIFFs exist, can reinterleave from parsed slice tiffs 

fprintf('Loading paramspath... %s\n', paramspath)
params = loadjson(paramspath);
A = loadjson(refpath);

source = params.PARAMS.preprocessing.sourcedir;
dest = params.PARAMS.preprocessing.destdir;

[processdir, childdir, ~] = fileparts(params.PARAMS.preprocessing.sourcedir);
simeta_fn = sprintf('SI_%s.json', params.PARAMS.source.run);
fprintf('SI: %s\n', fullfile(source, simeta_fn));
if ~exist(fullfile(source, simeta_fn), 'file')  % SI meta info is in raw, and we're trying to look in a processed dir...
    pparts = strsplit(source, '/processed');
    run_dir = pparts{1};
    raw_dirs = dir(fullfile(run_dir, 'raw*'));
    raw_dir = raw_dirs(1).name;    
    source = fullfile(run_dir, raw_dir);
    fprintf('Getting SI meta info from raw, path: %s', source);
end
simeta = loadjson(fullfile(source, simeta_fn));
%simeta = loadjson(A.raw_simeta_path);

fprintf('Running full bidi correction.\n')

namingFunction = @defaultNamingFunction;

nchannels = A.nchannels;
nslices = length(A.slices);
nvolumes = A.nvolumes;

% Check whether flyback corrected, or not:
if params.PARAMS.preprocessing.correct_flyback
    fprintf('Flyback corrected, not adjusting nslices.\n')'
    ndiscard = 0;
else
    % Check if there are any discarded frames:
    tmpfiles = fieldnames(simeta);
    ndiscard = simeta.(tmpfiles{1}).SI.hFastZ.numDiscardFlybackFrames;
    fprintf('Adding %i discard frames to nslices.\n', ndiscard);
end
%nslices = ndiscard + nslices;


tiffs = dir(fullfile(source, '*.tif'));
tiffs = {tiffs(:).name}';

fprintf('Found %i TIFF files in source:\n  %s\n', length(tiffs), source);
fprintf('Starting BIDI correction...\n')

for tiff_idx = 1:length(tiffs)
    currfile = sprintf('File%03d', tiff_idx); 
    nvolumes = simeta.(currfile).SI.hFastZ.numVolumes;
    nchannels = numel(simeta.(currfile).SI.hChannels.channelSave);
    nslices = simeta.(currfile).SI.hFastZ.numFramesPerVolume;
    fprintf('N slices per volume: %i\n', nslices);
    if nslices == length(A.slices) + ndiscard
        nslices = nslices - ndiscard
    end

    if nslices > 1 || nchannels > 1 %length(A.slices)>1 || A.nchannels>1
        do_deinterleave = true;
    else
        do_deinterleave = false;
    end
    
    if do_deinterleave
        write_dir_deinterleaved = sprintf('%s_deinterleaved', dest)
        fprintf('Writing deinterleaved files to: %s\n', write_dir_deinterleaved)
        if ~exist(write_dir_deinterleaved)
            mkdir(write_dir_deinterleaved)
        end
    end
    
    write_dir_interleaved = dest; %fullfile(mcparams.source_dir, mcparams.dest_dir);
    if ~exist(write_dir_interleaved)
        mkdir(write_dir_interleaved)
    end
    
    fprintf('Writing interleaved files to: %s\n', write_dir_interleaved)
    

    tpath = fullfile(source, tiffs{tiff_idx});
    fprintf('Processing tiff %i of %i...\n', tiff_idx, length(tiffs));
    [parent, filename_base, ext] = fileparts(tpath);

    %tic; Yt = read_file(tpath); toc; % is this faster
    currtiffpath = tpath;
    curr_file_name = sprintf('File%03d', tiff_idx);
    if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
        Yt = read_file(currtiffpath);
    else
        Yt = read_imgdata(currtiffpath);
    end
    [d1,d2,d3] = size(Yt);
    if d3 ~= nvolumes*nchannels*(nslices+ndiscard)
        fprintf('*** %s - Warning: fewer volumes found than specified...\n', curr_file_name) 
        fprintf('... N volumes: %i, N channels: %i, N slices %i (discard %i). Found %i.\n', nvolumes, nchannels, nslices, ndiscard, d3);
        %nvolumes = d3;
    end
    fprintf('Size loaded tiff: %s\n', mat2str(size(Yt)))
    [sx, sy, sz] = size(Yt);

    % Check filename for formatting:
    fi = strfind(filename_base, 'File');
    if isempty(fi)
        fidname = sprintf('File%03d', tiff_idx);
        fparts = strsplit(filename_base, '_');
        filename_base = strjoin([fparts(1:end-1), fidname], '_'); % ext];
        fid = tiff_idx;
    else
        fid = str2num(filename_base(fi+4:end));
    end

    % Either read every other channel from each tiff, or read each tiff
    % that is a single channel:       
    if params.PARAMS.preprocessing.split_channels
        fprintf('Channels are SPLIT.\n');
        fprintf('... Correcting TIFF: %s\n', filename_base);
        for channel_ix=1:nchannels
            fprintf('... %i of %i channels.', channel_ix, nchannels)'
            Yt_ch = Yt(:, :, channel_ix:nchannels:end);
            fprintf('... Single channel, mov size is: %s\n', mat2str(size(Yt_ch)));
            %nslices = size(Yt, 3)/nchannels/nvolumes
            Y = reshape(Yt_ch, [size(Yt_ch,1), size(Yt_ch,2), nslices+ndiscard, nvolumes]); 

            if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
            %Y = Y -  min(Y(:));                         % make data non-negative
            fprintf('... Correcting bidirectional scanning offset.\n');
            Ydata = correct_bidirectional_phasing(Y(:,:,1:nslices,:)); % only data frames
            Y(:,:,1:nslices,:) = Ydata;
            filename = sprintf('%s_Channel%02d', filename_base, channel_ix);
            fprintf('... Single channel filename: %s\n', filename);
            tiffWrite(Y, strcat(filename, ext), write_dir_interleaved, 'int16')
            fprintf('... Writing bidi-corrected tif to file: %s\n', filename)'
            if do_deinterleave % Also save deinterleaved:
                deinterleave_tiffs(Y, filename, fid, write_dir_deinterleaved, A);
            end
        end
    else
        fprintf('Channels are not split.\n');
        newtiff = zeros(d1,d2,(nslices+ndiscard)*nchannels*nvolumes);
	fprintf('... Size of interleaved tiff: %s\n', mat2str(size(newtiff)));
        fprintf('... Correcting TIFF: %s\n', filename_base); 
        if nchannels>1
            fprintf('... Grabbing every other channel.\n')
        else
            fprintf('... Only 1 channel.\n');
        end
        for cidx=1:nchannels
	    fprintf('... Processing channel %i of %i...\n', cidx, nchannels);
            Yt_ch = Yt(:,:,cidx:nchannels:end);
            fprintf('... Channel %i, mov size is: %s\n', cidx, mat2str(size(Yt_ch)));
            %nslices = size(Yt_ch, 3)/nvolumes
            Y = reshape(Yt_ch, [size(Yt_ch,1), size(Yt_ch,2), nslices+ndiscard, nvolumes]); 

            if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
            %Y = Y -  min(Y(:));                         % make data non-negative
            fprintf('... Reinterleaved TIFF (single ch) size: %s\n', mat2str(size(Y)))
            fprintf('... Correcting bidirectional scanning offset.\n');
            Ydata = correct_bidirectional_phasing(Y(:,:,1:nslices,:));
	    Y(:,:,1:nslices,:) = Ydata;
            Y = reshape(Y, [size(Yt_ch,1), size(Yt,2), (nslices+ndiscard)*nvolumes]); 
            newtiff(:,:,cidx:nchannels:end) = Y;
            clear Y; clear Yt_ch;
        end
        tiffWrite(newtiff, strcat(filename_base, ext), write_dir_interleaved, 'int16')
        fprintf('... Interleaved tif saved to: %s\n', strcat(filename_base, ext)); 
        if do_deinterleave % Also save deinterleaved:
            deinterleave_tiffs(newtiff, filename_base, fid, write_dir_deinterleaved, A);
        end
 
    end
    fprintf('***** Finished bidi-correction. *****\n');
end
  

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end

end
