function simeta = get_scan_info(A)
        
        % Load mcparam info:
        load(A.mcparams_path);
 
        tiff_source = mcparams.tiff_dir;
        acquisition_name = mcparams.acquisition_name;
        simeta = struct();
        
        metadata_fn = sprintf('%s.mat', acquisition_name)
        metadata = load(fullfile(tiff_source, metadata_fn));
        
        ntiffs = length(metadata.(acquisition_name).metaDataSI);
        sistruct = struct();  
        for fidx=1:ntiffs
            curr_meta = metadata.(acquisition_name).metaDataSI{fidx};
            
            % Sort Parsed files into separate directories if needed:
            nChannels = length(curr_meta.SI.hChannels.channelSave);            
            nVolumes = curr_meta.SI.hFastZ.numVolumes;

            nSlices = curr_meta.SI.hFastZ.numFramesPerVolume;
            nDiscard = curr_meta.SI.hFastZ.numDiscardFlybackFrames;
            nFramesPerVolume = nSlices; % + nDiscard;
            nTotalFrames = nFramesPerVolume * nVolumes;
            
            siFrameTimes = curr_meta.frameTimestamps_sec(1:nChannels:end);
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
            sistruct.file(fidx).siFrameTimes = siFrameTimes;
            
            sistruct.file(fidx).siFrameRate = siFrameRate;
            sistruct.file(fidx).siVolumeRate = siVolumeRate;
            sistruct.file(fidx).frameWidth = frameWidth;
            sistruct.file(fidx).slowMultiplier = slowMultiplier;
            sistruct.file(fidx).linesPerFrame = linesPerFrame;
            sistruct.file(fidx).frameHeight = frameHeight;
            if isfield(metadata.(acquisition_name), 'motionRefMovNum')
                motionRefNum = metadata.(acquisition_name).motionRefMovNum;
                motionRefPath = metadata.(acquisition_name).Movies{motionRefNum};
            
                sistruct.file(fidx).motionRefNum = motionRefNum;
                sistruct.file(fidx).motionRefPath = motionRefPath;
            
                sistruct.file(fidx).rawTiffPath = metadata.(acquisition_name).Movies{fidx};
            %sistruct.file(fidx) = sistruct;
            end
            
        end
        simeta.acquisition_name = acquisition_name;
        simeta.ntiffs = ntiffs;
        simeta.nchannels = nChannels;
        simeta.SI = sistruct;

end
