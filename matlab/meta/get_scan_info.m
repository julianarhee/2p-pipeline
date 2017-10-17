function simeta = get_scan_info(I, A)

% TODO:  Replace this tmp func with metadata info extracted from raw .mat (or .json) from Step1 (process_raw.py)
        
        % Load mcparam info:
        mcparams = load(A.mcparams_path);
        mcparams = mcparams.(I.mc_id)
 
        tiff_source = mcparams.tiff_dir;
        base_filename = A.base_filename; % TODO: again, make sure this isn't specific to mcparms.method
        simeta = struct();
        
        metadata_fn = sprintf('%s.mat', base_filename)
        metadata = load(fullfile(tiff_source, metadata_fn));
        
        ntiffs = length(metadata.(base_filename).metaDataSI);
        sistruct = struct();  
        for fidx=1:ntiffs
            %curr_meta = metadata.(base_filename).metaDataSI{fidx};
            curr_meta = metadata.(base_filename).metaDataSI{fidx}; %.SI;
            curr_meta
            % Sort Parsed files into separate directories if needed:
            curr_meta.SI.hChannels
            nChannels = length(curr_meta.SI.hChannels.channelSave);             
            nVolumes = curr_meta.SI.hFastZ.numVolumes;

            nSlices = curr_meta.SI.hFastZ.numFramesPerVolume;
            nDiscard = curr_meta.SI.hFastZ.numDiscardFlybackFrames;
            nFramesPerVolume = nSlices; % + nDiscard;
            nTotalFrames = nFramesPerVolume * nVolumes;
            
            %siFrameTimes = curr_meta.frameTimestamps_sec(1:nChannels:end);
            siFrameRate = curr_meta.SI.hRoiManager.scanFrameRate;
            siVolumeRate = curr_meta.SI.hRoiManager.scanVolumeRate;

            frameWidth = curr_meta.SI.hRoiManager.pixelsPerLine;
            slowMultiplier = curr_meta.SI.hRoiManager.scanAngleMultiplierSlow;
            linesPerFrame = curr_meta.SI.hRoiManager.linesPerFrame;
            frameHeight = linesPerFrame; %linesPerFrame/slowMultiplier

            sistruct.file(fidx).nChannels = nChannels;
            sistruct.file(fidx).nVolumes = nVolumes;

            sistruct.file(fidx).nSlices = nSlices - nDiscard;
            sistruct.file(fidx).nDiscard = nDiscard;
            sistruct.file(fidx).nFramesPerVolume = nFramesPerVolume;
            sistruct.file(fidx).nTotalFrames = nTotalFrames;
            %sistruct.file(fidx).siFrameTimes = siFrameTimes;
            
            sistruct.file(fidx).siFrameRate = siFrameRate;
            sistruct.file(fidx).siVolumeRate = siVolumeRate;
            sistruct.file(fidx).frameWidth = frameWidth;
            sistruct.file(fidx).slowMultiplier = slowMultiplier;
            sistruct.file(fidx).linesPerFrame = linesPerFrame;
            sistruct.file(fidx).frameHeight = frameHeight;
            if isfield(metadata.(base_filename), 'motionRefMovNum')
                motionRefNum = metadata.(base_filename).motionRefMovNum;
                motionRefPath = metadata.(base_filename).Movies{motionRefNum};
            
                sistruct.file(fidx).motionRefNum = motionRefNum;
                sistruct.file(fidx).motionRefPath = motionRefPath;
            
                sistruct.file(fidx).rawTiffPath = metadata.(base_filename).Movies{fidx};
            %sistruct.file(fidx) = sistruct;
            end
            
        end
        simeta.base_filename = base_filename;
        simeta.ntiffs = ntiffs;
        simeta.nchannels = nChannels;
        simeta.SI = sistruct;

end
