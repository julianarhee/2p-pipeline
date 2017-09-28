function do_bidi_correction(mcparams)

namingFunction = @defaultNamingFunction;
write_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);

% Load SI meta info (from MC, for now) to get TIFF info:
switch mcparams.method
    case 'Acquisition2P'
        tmp_acq_obj = load(mcparams.info.acq_object_path);
        acq_obj = tmp_acq_obj.(mcparams.info.acquisition_name);
        clear tmp_acq_obj
    case 'NoRMCorre'
        % TODO:  do this
    otherwise
        % TODO:  need a way to get meta info     
end

nchannels = mcparams.nchannels;
nslices = acq_obj.metaDataSI{1}.SI.hFastZ.numFramesPerVolume;
nvolumes = acq_obj.metaDataSI{1}.SI.hFastZ.numVolumes;

% Grab (corrected) TIFFs from DATA (or acquisition) dir for correction:
tiffs = dir(fullfile(mcparams.tiff_dir, '*.tif'));
fprintf('Doing bidi-correction on TIFFs in dir: %s', mcparams.tiff_dir);
tiff_dir = mcparams.tiff_dir; %D.dataDir;
tiffs = {tiffs(:).name}'


for tiff_idx = 1:length(tiffs)
           
    tpath = fullfile(tiff_dir, tiffs{tiff_idx});
    fprintf('Processing tiff from path:\n%s\n', tpath);
    [source, filename, ext] = fileparts(tpath);

    tic; Yt = read_file(tpath); toc; % is this faster
    [d1,d2,d3] = size(Yt);
    fprintf('Size slice t-series tiff: %s\n', mat2str(size(Yt)))
    fi = strfind(filename, 'File');
    fid = str2num(filename(fi+6));


    % Either read every other channel from each tiff, or read each tiff
    % that is a single channel:
    if mcparams.flyback_corrected && ~mcparams.split_channels
        newtiff = zeros(d1,d2,nslices*nchannels*nvolumes);
        fprintf('Correcting TIFF: %s\n', filename); 
        fprintf('Grabbing every other channel.\n')
        for cidx=1:mcparams.nchannels
            Yt_ch = Yt(:,:,cidx:nchannels:end);
            fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt_ch)));
            Y = cell(1, nvolumes);
            firstslice = 1; %startSliceIdx; %1;
            for vol=1:nvolumes
                Y{vol} = Yt_ch(:,:,firstslice:(firstslice+nslices-1));
                firstslice = firstslice+nslices;
            end
            Y = cat(4, Y{1:end});
            if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
            Y = Y -  min(Y(:));                         % make data non-negative
            fprintf('Correcting bidirectional scanning offset.\n');
            Y = correct_bidirectional_phasing(Y);
            newtiff(:,:,cidx:nchannels:end) = Y;
            clearvars Y Yt_ch
        end
        tiffWrite(newtiff, strcat(filename, '.tif'), source)
        
        % Also save deinterleaved:
        fprintf('Saving deinterleaved slices to:\n%s\n', write_dir);
        for sl = 1:nslices
            for ch = 1:nchannels
                frame_idx = ch + (sl-1)*nchannels;
                
                % Create movie fileName and save to default format
                % TODO: set this to work with other mc methods....
                if strcmp(mcparams.method, 'Acquisition2P')
                    mov_filename = feval(namingFunction,mcparams.info.acquisition_name, sl, ch, fid);
                    try
                        tiffWrite(newtiff(:, :, frame_idx:(nslices*nchannels):end), mov_filename, write_dir);
                    catch
                        % Sometimes, disk access fails due to intermittent
                        % network problem. In that case, wait and re-try once:
                        pause(60);
                        tiffWrite(newtiff(:, :, frame_idx:(nslices*nchannels):end), mov_filename, write_dir);
                    end
                end
            end
        end
                
    elseif mcparams.flyback_corrected && mcparams.split_channels
	fprintf('Correcting TIFF: %s\n', filename);
        fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt)));
        Y = cell(1, nvolumes);
        firstslice = 1; %startSliceIdx; %1;
        for vol=1:nvolumes
            Y{vol} = Yt(:,:,firstslice:(firstslice+nslices-1));
            firstslice = firstslice+nslices;
        end
        Y = cat(4, Y{1:end});
        if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
        Y = Y -  min(Y(:));                         % make data non-negative
        fprintf('Correcting bidirectional scanning offset.\n');
        Y = correct_bidirectional_phasing(Y);
        tiffWrite(Y, strcat(filename, '.tif'), source)
        
        % Also save deinterleaved:
        fprintf('Saving deinterleaved slices to:\n%s\n', write_dir);
        for sl = 1:nslices
            
            frame_idx = sl; %1 + (sl-1)*nchannels;
            
            % Create movie fileName and save to default format
            if strfind(filename, 'Channel01')
                ch=1;
            elseif strfind(filename, 'Channel02')
                ch=2;
            end
            
            % Create movie fileName and save to default format
            % TODO: set this to work with other mc methods....
            if strcmp(mcparams.method, 'Acquisition2P')
                mov_filename = feval(namingFunction,mcparams.info.acquisition_name, sl, ch, fid);
                try
                    tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir);
                    tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir);
                catch
                    % Sometimes, disk access fails due to intermittent
                    % network problem. In that case, wait and re-try once:
                    pause(60);
                    tiffWrite(Y(:, :, frame_idx:(nslices):end), mov_filename, write_dir);
                end
            end
            
        end
        
    end

end
   

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end

end
