function S = get_si_info(D)
        
        %sourceDir = D.sourceDir;
        sourceDir = D.dataDir;
        acquisitionName = D.acquisitionName;
        siMetaStruct = struct();
        
        siMetaName = sprintf('%s.mat', acquisitionName);
        meta = load(fullfile(sourceDir, siMetaName));
        
        if D.metaonly && isfield(D,'nTiffs')
            nTiffs = D.nTiffs;
        else
            nTiffs = length(meta.(acquisitionName).metaDataSI)
        end
        
        for fidx=1:nTiffs
            currMeta = meta.(acquisitionName).metaDataSI{fidx};
            
            % Sort Parsed files into separate directories if needed:
            nChannels = length(currMeta.SI.hChannels.channelSave);            
            nVolumes = currMeta.SI.hFastZ.numVolumes;

            nSlices = currMeta.SI.hFastZ.numFramesPerVolume;
            nDiscard = currMeta.SI.hFastZ.numDiscardFlybackFrames;
            nFramesPerVolume = nSlices; % + nDiscard;
            nTotalFrames = nFramesPerVolume * nVolumes;

            siFrameTimes = currMeta.frameTimestamps_sec(1:2:end);
            siFrameRate = currMeta.SI.hRoiManager.scanFrameRate;
            siVolumeRate = currMeta.SI.hRoiManager.scanVolumeRate;

            frameWidth = currMeta.SI.hRoiManager.pixelsPerLine;
            slowMultiplier = currMeta.SI.hRoiManager.scanAngleMultiplierSlow;
            linesPerFrame = currMeta.SI.hRoiManager.linesPerFrame;
            frameHeight = linesPerFrame; %linesPerFrame/slowMultiplier

            siMetaStruct.file(fidx).nChannels = nChannels;
            siMetaStruct.file(fidx).nVolumes = nVolumes;

            siMetaStruct.file(fidx).nSlices = nSlices - nDiscard;
            siMetaStruct.file(fidx).nDiscard = nDiscard;
            siMetaStruct.file(fidx).nFramesPerVolume = nFramesPerVolume;
            siMetaStruct.file(fidx).nTotalFrames = nTotalFrames;
            siMetaStruct.file(fidx).siFrameTimes = siFrameTimes;
            
            siMetaStruct.file(fidx).siFrameRate = siFrameRate;
            siMetaStruct.file(fidx).siVolumeRate = siVolumeRate;
            siMetaStruct.file(fidx).frameWidth = frameWidth;
            siMetaStruct.file(fidx).slowMultiplier = slowMultiplier;
            siMetaStruct.file(fidx).linesPerFrame = linesPerFrame;
            siMetaStruct.file(fidx).frameHeight = frameHeight;
            if isfield(meta.(acquisitionName), 'motionRefMovNum')
                motionRefNum = meta.(acquisitionName).motionRefMovNum;
                motionRefPath = meta.(acquisitionName).Movies{motionRefNum};
            
                siMetaStruct.file(fidx).motionRefNum = motionRefNum;
                siMetaStruct.file(fidx).motionRefPath = motionRefPath;
            
                siMetaStruct.file(fidx).rawTiffPath = meta.(acquisitionName).Movies{fidx};
            %siMetaStruct.file(fidx) = sistruct;
            end
            
        end
        S.acquisitionName = acquisitionName;
        S.nTiffs = nTiffs;
        S.nChannels = nChannels;
        S.SI = siMetaStruct;

end
