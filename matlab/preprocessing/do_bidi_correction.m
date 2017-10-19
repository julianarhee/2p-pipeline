function mcparams = do_bidi_correction(source, mcparams, A, from_deinterleaved)

% Reads in TIFFs and does correction for bidirection-scanning.
% For each TIFF in acquisition, does correction on interleaved tiff, saves to dir:  <source>_Bidi
% Also saves deinterleaved tiffs to dir:  <source>_Bidi_slices dir

% INPUTs:
% source - can be 'Corrected' or 'Parsed' (i.e., do correcitonon mc or raw data)
% varargin - if no interleaved TIFFs exist, can reinterleave from parsed slice tiffs 

simeta = load(A.raw_simeta_path);

if ~exist('from_deinterleaved')
    from_deinterleaved = false;
end

if mcparams.bidi_corrected
    mcparams.dest_dir = sprintf('%s_Bidi', source); %'Corrected_Bidi';
    if ~exist(fullfile(mcparams.source_dir, mcparams.dest_dir), 'dir')
        mkdir(fullfile(mcparams.source_dir, mcparams.dest_dir));
    end
end

namingFunction = @defaultNamingFunction;

nchannels = A.nchannels;
nslices = length(A.slices);
nvolumes = A.nvolumes;
 
% Grab (corrected) TIFFs from DATA (or acquisition) dir for correction:
tiffs = dir(fullfile(mcparams.source_dir, '*.tif'));
tiff_dir = mcparams.source_dir; %D.dataDir;
tiffs = {tiffs(:).name}'
if length(tiffs)==0
    tiff_dir = fullfile(mcparams.source_dir, source);
%     if mcparams.corrected
%         tiff_dir = fullfile(mcparams.source_dir, 'Corrected');
%     else
%         tiff_dir = fullfile(mcparams.source_dir, 'Raw');
%     end
    tiffs = dir(fullfile(tiff_dir, '*.tif'));
    tiffs = {tiffs(:).name}';
end

fprintf('Found %i TIFF files.\n', length(tiffs));

%write_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
write_dir_deinterleaved = fullfile(mcparams.source_dir, sprintf('%s_slices', mcparams.dest_dir));
if ~exist(write_dir_deinterleaved)
    mkdir(write_dir_deinterleaved)
end
write_dir_interleaved = fullfile(mcparams.source_dir, mcparams.dest_dir);
if ~exist(write_dir_interleaved)
    mkdir(write_dir_interleaved)
end

fprintf('Starting BIDI correction...\n')
fprintf('Writing deinterleaved files to: %s\n', write_dir_deinterleaved)
fprintf('Writing interleaved files to: %s\n', write_dir_interleaved)



for tiff_idx = 1:length(tiffs)
    currfile = sprintf('File%03d', tiff_idx); 
    nvolumes = simeta.(currfile).SI.hFastZ.numVolumes;
    nchannels = mcparams.nchannels;
 
    tpath = fullfile(tiff_dir, tiffs{tiff_idx});
    fprintf('Processing tiff %i of %i...\n', tiff_idx, length(tiffs));
    [source, filename, ext] = fileparts(tpath);

    if from_deinterleaved
        fprintf('Reinterleaving tiffs from source for BiDi correction...\n');
        slice_dir = fullfile(mcparams.source_dir, preprocessing_source); 
        tic; Yt = reinterleave_from_source(slice_dir, A); toc;
        fprintf('Got reinterleaved TIFF from source. Size is: %s\n', mat2str(size(Yt)));
    else
        %tic; Yt = read_file(tpath); toc; % is this faster
        currtiffpath = tpath;
        curr_file_name = sprintf('File%03d', tiff_idx);
        if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
            Yt = read_file(currtiffpath);
        else
            Yt = read_imgdata(currtiffpath);
        end

    end
    [d1,d2,~] = size(Yt);
    fprintf('Size interleaved tiff: %s\n', mat2str(size(Yt)))
    fi = strfind(filename, 'File');
    %fid = str2num(filename(fi+6));
    fid = str2num(filename(fi+4:end));

    % Either read every other channel from each tiff, or read each tiff
    % that is a single channel:
    %if mcparams.flyback_corrected && ~mcparams.split_channels
    if ~mcparams.split_channels

        newtiff = zeros(d1,d2,nslices*nchannels*nvolumes);
        fprintf('Correcting TIFF: %s\n', filename); 
        fprintf('Grabbing every other channel.\n')
        for cidx=1:mcparams.nchannels
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
        
        % Also save deinterleaved:
        deinterleave_tiffs(newtiff, filename, fid, write_dir_deinterleaved, A);
                 
    elseif mcparams.split_channels
	    fprintf('Correcting TIFF: %s\n', filename);
        fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt)));
        nslices = size(Yt, 3)/nchannels/nvolumes
        Y = reshape(Yt, [size(Yt,1), size(Yt,2), nslices, nvolumes]); 

        if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
        %Y = Y -  min(Y(:));                         % make data non-negative
        fprintf('Correcting bidirectional scanning offset.\n');
        Y = correct_bidirectional_phasing(Y);
        tiffWrite(Y, strcat(filename, '.tif'), write_dir_interleaved, 'int16')
        
        % Also save deinterleaved:
        deinterleave_tiffs(Y, filename, fid, write_dir_deinterleaved, A);         
    end

end

fprintf('Finished bidi-correction.\n');

save(fullfile(mcparams.source_dir, 'mcparams.mat'), 'mcparams', '-append');

  

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end

end
