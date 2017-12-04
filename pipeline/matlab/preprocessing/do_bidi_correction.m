function do_bidi_correction(paramspath, refpath)

% Reads in TIFFs and does correction for bidirection-scanning.
% For each TIFF in acquisition, does correction on interleaved tiff, saves to dir:  <source>_Bidi
% Also saves deinterleaved tiffs to dir:  <source>_Bidi_slices dir

% INPUTs:
% source - can be 'Corrected' or 'Parsed' (i.e., do correcitonon mc or raw data)
% varargin - if no interleaved TIFFs exist, can reinterleave from parsed slice tiffs 

params = loadjson(paramspath);
A = loadjson(refpath);

source = params.PARAMS.preprocessing.sourcedir;
dest = params.PARAMS.preprocessing.destdir;

[processdir, childdir, ~] = fileparts(params.PARAMS.preprocessing.sourcedir);
simeta_fn = sprintf('SI_%s.json', params.PARAMS.source.run);
fprintf('SI: %s\n', fullfile(source, simeta_fn));
simeta = loadjson(fullfile(source, simeta_fn));
%simeta = loadjson(A.raw_simeta_path);

fprintf('Running full bidi correction.\n')

namingFunction = @defaultNamingFunction;

nchannels = A.nchannels;
nslices = length(A.slices);
nvolumes = A.nvolumes;

tiffs = dir(fullfile(source, '*.tif'));
tiffs = {tiffs(:).name}';

fprintf('Found %i TIFF files in source:\n  %s\n', length(tiffs), source);

if length(A.slices)>1 || A.nchannels>1
    do_deinterleave = true;
else
    do_deinterleave = false;
end

if do_deinterleave
    write_dir_deinterleaved = sprintf('%s_slices', dest)
    fprintf('Writing deinterleaved files to: %s\n', write_dir_deinterleaved)
    if ~exist(write_dir_deinterleaved)
        mkdir(write_dir_deinterleaved)
    end
end

write_dir_interleaved = dest; %fullfile(mcparams.source_dir, mcparams.dest_dir);
if ~exist(write_dir_interleaved)
    mkdir(write_dir_interleaved)
end

fprintf('Starting BIDI correction...\n')
fprintf('Writing interleaved files to: %s\n', write_dir_interleaved)

for tiff_idx = 1:length(tiffs)
    currfile = sprintf('File%03d', tiff_idx); 
    nvolumes = simeta.(currfile).SI.hFastZ.numVolumes;
 
    tpath = fullfile(source, tiffs{tiff_idx});
    fprintf('Processing tiff %i of %i...\n', tiff_idx, length(tiffs));
    [parent, filename, ext] = fileparts(tpath);

    %tic; Yt = read_file(tpath); toc; % is this faster
    currtiffpath = tpath;
    curr_file_name = sprintf('File%03d', tiff_idx);
    if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
        Yt = read_file(currtiffpath);
    else
        Yt = read_imgdata(currtiffpath);
    end
    [d1,d2,~] = size(Yt);
    fprintf('Size interleaved tiff: %s\n', mat2str(size(Yt)))
    fi = strfind(filename, 'File');
    fid = str2num(filename(fi+4:end));

    % Either read every other channel from each tiff, or read each tiff
    % that is a single channel:       
    if params.PARAMS.preprocessing.split_channels
        fprintf('Correcting TIFF: %s\n', filename);
        fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt)));
        nslices = size(Yt, 3)/nchannels/nvolumes
        Y = reshape(Yt, [size(Yt,1), size(Yt,2), nslices, nvolumes]); 

        if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
        %Y = Y -  min(Y(:));                         % make data non-negative
        fprintf('Correcting bidirectional scanning offset.\n');
        Y = correct_bidirectional_phasing(Y);
        tiffWrite(Y, strcat(filename, '.tif'), write_dir_interleaved, 'int16')

        if do_deinterleave % Also save deinterleaved:
            deinterleave_tiffs(Y, filename, fid, write_dir_deinterleaved, A);                 end
    else
        newtiff = zeros(d1,d2,nslices*nchannels*nvolumes);
        fprintf('Correcting TIFF: %s\n', filename); 
        if nchannels>1
            fprintf('Grabbing every other channel.\n')
        else
            fprintf('Not splitting channels.\n');
        end
        for cidx=1:nchannels
            Yt_ch = Yt(:,:,cidx:nchannels:end);
            fprintf('Channel %i, mov size is: %s\n', cidx, mat2str(size(Yt_ch)));
            nslices = size(Yt_ch, 3)/nvolumes
            Y = reshape(Yt_ch, [size(Yt_ch,1), size(Yt_ch,2), nslices, nvolumes]); 

            if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
            %Y = Y -  min(Y(:));                         % make data non-negative
            fprintf('Reinterleaved TIFF (single ch) size: %s\n', mat2str(size(Y)))
            fprintf('Correcting bidirectional scanning offset.\n');
            [Y, ~] = correct_bidirectional_phasing(Y);
            Y = reshape(Y, [size(Yt_ch,1), size(Yt,2), nslices*nvolumes]); 
            newtiff(:,:,cidx:nchannels:end) = Y;
            clearvars Y Yt_ch
        end
        tiffWrite(newtiff, strcat(filename, '.tif'), write_dir_interleaved, 'int16')
        
        if do_deinterleave % Also save deinterleaved:
            deinterleave_tiffs(newtiff, filename, fid, write_dir_deinterleaved, A);
        end
 
    end
    fprintf('Finished bidi-correction.\n');
end
  

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end

end
