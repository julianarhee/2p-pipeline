function [A, mcparams] = do_bidi_correction(A, mcparams, varargin)

% If provided opt arg, do Bi-Di correction from deinterleaved TIFFs.
% Otherwise, read TIFFs in ./DATA dir (output from first MC + reinterleaving step.
% Output of BiDi correction on interleaved TIFFs will be written to ./DATA (overwrites).
% Deinterleaved files saved to ./DATA/Corrected_Bidi. No additional reinterleaving step needed.

simeta = load(A.raw_simeta_path);

if length(varargin)>0
    path_to_parsed = varargin{1};
    from_source = true;
else
    from_source = false;
end

if A.use_bidi_corrected
mcparams.bidi_corrected_dir = 'Corrected_Bidi';
if ~exist(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir), 'dir')
    mkdir(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir));
end
end

namingFunction = @defaultNamingFunction;
write_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);

nchannels = A.nchannels;
nslices = length(A.slices);
nvolumes = A.nvolumes;
 
% Grab (corrected) TIFFs from DATA (or acquisition) dir for correction:
tiffs = dir(fullfile(mcparams.tiff_dir, '*.tif'));
fprintf('Doing bidi-correction on TIFFs in dir: %s', mcparams.tiff_dir);
tiff_dir = mcparams.tiff_dir; %D.dataDir;
tiffs = {tiffs(:).name}'
fprintf('Found %i TIFF files.\n', length(tiffs));

for tiff_idx = 1:length(tiffs)
           
    tpath = fullfile(tiff_dir, tiffs{tiff_idx});
    fprintf('Processing tiff %i of %i...\n', tiff_idx, length(tiffs));
    [source, filename, ext] = fileparts(tpath);

    if from_source
        curr_filedir = sprintf('File%03d', tiff_idx);
        fprintf('Reinterleaving tiffs from source for BiDi correction...\n');
        tic; Yt = reinterleave_from_source(curr_filedir, path_to_parsed, A); toc;
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
    if mcparams.flyback_corrected && ~mcparams.split_channels
        newtiff = zeros(d1,d2,nslices*nchannels*nvolumes);
        fprintf('Correcting TIFF: %s\n', filename); 
        fprintf('Grabbing every other channel.\n')
        for cidx=1:mcparams.nchannels
            Yt_ch = Yt(:,:,cidx:nchannels:end);
            fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt_ch)));
            Y = reshape(Yt_ch, [size(Yt_ch,1), size(Yt_ch,2), nslices, nvolumes]); 

            if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
            Y = Y -  min(Y(:));                         % make data non-negative
            fprintf('Reinterleaved TIFF (single ch) size: %s\n', mat2str(size(Y)))
            fprintf('Correcting bidirectional scanning offset.\n');
            [Y, ~] = correct_bidirectional_phasing(Y);
            Y = reshape(Y, [size(Yt_ch,1), size(Yt,2), nslices*nvolumes]); 
            newtiff(:,:,cidx:nchannels:end) = Y;
            clearvars Y Yt_ch
        end
        tiffWrite(newtiff, strcat(filename, '.tif'), source, 'int16')
        
        % Also save deinterleaved:
        deinterleave_tiffs(newtiff, filename, fid, write_dir, A);
                 
    elseif mcparams.flyback_corrected && mcparams.split_channels
	    fprintf('Correcting TIFF: %s\n', filename);
        fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt)));
        Y = reshape(Yt, [size(Yt,1), size(Yt,2), nslices, nvolumes]); 

        if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
        Y = Y -  min(Y(:));                         % make data non-negative
        fprintf('Correcting bidirectional scanning offset.\n');
        Y = correct_bidirectional_phasing(Y);
        tiffWrite(Y, strcat(filename, '.tif'), source, 'int16')
        
        % Also save deinterleaved:
        deinterleave_tiffs(Y, filename, fid, write_dir, A);         
    end

end

fprintf('Finished bidi-correction.\n');

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');

  

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end

end
