function mcparams = do_bidi_correction(mcparams, A, from_deinterleaved)

% Reads in TIFFs and does correction for bidirection-scanning.
% For each TIFF in acquisition, does correction on interleaved tiff, saves to dir:  <source>_Bidi
% Also saves deinterleaved tiffs to dir:  <source>_Bidi_slices dir

% INPUTs:
% source - can be 'Corrected' or 'Parsed' (i.e., do correcitonon mc or raw data)
% varargin - if no interleaved TIFFs exist, can reinterleave from parsed slice tiffs 


source = mcparams.dest_dir;

simeta = load(A.raw_simeta_path);

if ~exist('from_deinterleaved')
    from_deinterleaved = false;
end

if mcparams.bidi_corrected
    if ~any(strfind(mcparams.dest_dir, 'Bidi'))
        mcparams.dest_dir = sprintf('%s_Bidi', mcparams.dest_dir); %'Corrected_Bidi';
    end
    if ~exist(fullfile(mcparams.source_dir, mcparams.dest_dir), 'dir')
        mkdir(fullfile(mcparams.source_dir, mcparams.dest_dir)); 
        do_bidi = true;
    else 
        % Check if BIDI already done:
        bidi_output_dir = fullfile(mcparams.source_dir, mcparams.dest_dir);
        if length(A.slices)>1 || A.nchannels>1
            nchannel_dirs = dir(fullfile(bidi_output_dir, '*Channel*'));
            if length(nchannel_dirs)==A.nchannels
                nfile_dirs = dir(fullfile(bidi_output_dir, nchannel_dirs(1).name, '*File*'));
                if length(nfile_dirs)==A.ntiffs
                    tiffs = dir(fullfile(bidi_output_dir, nchannel_dirs(1).name, nfile_dirs(1).name, '*.tif'));
                    if length(tiffs)==length(A.slices)
                        found_correct_ntiffs = true;
                    else
                        found_correct_ntiffs = false;
                    end
                else
                    found_correct_ntiffs = false;
                end
            else
                found_correct_ntiffs = false; 
            end
        else
            % SINGLE channel, SINGLE file (no parsed folders)
            tiffs = dir(fullfile(bidi_output_dir, '*.tif'))
            if length(tiffs)==A.ntiffs
                found_correct_ntiffs = true;
            else
                found_correct_ntiffs = false;
            end
        end

        if found_correct_ntiffs 
            fprintf('Found correct num BIDI-corrected TIFFs in dir %s\n', mcparams.dest_dir);
            user_says_bidi_again = input('Press Y/n to re-run bidi correction: ', 's');
            if strcmp(user_says_bidi_again, 'Y')
                do_bidi = true;
            else
                do_bidi = false;
            end
        else
            do_bidi = true;
        end 
    end
end

if ~do_bidi
    fprintf('NOT re-doing bidi-correction.\n');
else
    fprintf('Running full bidi correction.\n')

    namingFunction = @defaultNamingFunction;

    nchannels = A.nchannels;
    nslices = length(A.slices);
    nvolumes = A.nvolumes;
     
    % Grab (corrected) TIFFs from DATA (or acquisition) dir for correction:
    while (1)
        tiff_source = fullfile(mcparams.source_dir, source)
        tiffs = dir(fullfile(tiff_source, '*.tif'));
        tiffs = {tiffs(:).name}'
        if length(tiffs)==0
            fprintf('No TIFFs found in specified bidi source dir:\n%s\n', tiff_source);
            source = input('Enter folder name of tiffs to bidi-correct:\n', 's');
        else
            break; 
        end
    end

    fprintf('Found %i TIFF files in source:\n  %s\n', length(tiffs), tiff_source);

    %write_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
    if length(A.slices)>1 || A.nchannels>1
        fprintf('Writing deinterleaved files to: %s\n', write_dir_deinterleaved)
        write_dir_deinterleaved = fullfile(mcparams.source_dir, sprintf('%s_slices', mcparams.dest_dir));
        if ~exist(write_dir_deinterleaved)
            mkdir(write_dir_deinterleaved)
        end
    end

    write_dir_interleaved = fullfile(mcparams.source_dir, mcparams.dest_dir);
    if ~exist(write_dir_interleaved)
        mkdir(write_dir_interleaved)
    end

    fprintf('Starting BIDI correction...\n')
    fprintf('Writing interleaved files to: %s\n', write_dir_interleaved)


    for tiff_idx = 1:length(tiffs)
        currfile = sprintf('File%03d', tiff_idx); 
        nvolumes = simeta.(currfile).SI.hFastZ.numVolumes;
        nchannels = mcparams.nchannels;
     
        tpath = fullfile(tiff_source, tiffs{tiff_idx});
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
        fid = str2num(filename(fi+4:end));

        % Either read every other channel from each tiff, or read each tiff
        % that is a single channel:
        if ~mcparams.split_channels

            newtiff = zeros(d1,d2,nslices*nchannels*nvolumes);
            fprintf('Correcting TIFF: %s\n', filename); 
            if mcparams.nchannels>1
                fprintf('Grabbing every other channel.\n')
            else
                fprintf('Not splitting channels.\n');
            end
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
            
            if length(A.slices)>1 || A.nchannels>1 
                % Also save deinterleaved:
                deinterleave_tiffs(newtiff, filename, fid, write_dir_deinterleaved, A);
            end
             
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

            if length(A.slices)>1 || A.nchannels>1        
                % Also save deinterleaved:
                deinterleave_tiffs(Y, filename, fid, write_dir_deinterleaved, A);         
            end
        end

    end

    fprintf('Finished bidi-correction.\n');

    %save(fullfile(mcparams.source_dir, 'mcparams.mat'), 'mcparams', '-append');
end
  

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end

end
