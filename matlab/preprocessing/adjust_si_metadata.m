function scanImageMetadata = adjust_si_metadata(scanImageMetadata, mov_size)

% For pre-processed substacks, adjust meta data to match substack
% TIFFs that have non-data frames (artefact/bad flyback frames, discarded
% flyback frames, etc.) removed.

fprintf('Parsing processed SI tiff and getting adjusted meta data...\n');
fprintf('Size of movie: %s\n', mat2str(mov_size)); %mat2str(size(mov)));

nSlicesTmp = scanImageMetadata.SI.hStackManager.numSlices
nDiscardTmp = scanImageMetadata.SI.hFastZ.numDiscardFlybackFrames
nVolumesTmp = scanImageMetadata.SI.hFastZ.numVolumes
nChannelsTmp = length(scanImageMetadata.SI.hChannels.channelSave)
% nChannelsTmp = scanImageMetadata.SI.hChannels.channelSave
% if length(nChannelsTmp)>3
%     nChannelsTmp = length(str2num(scanImageMetadata.SI.hChannels.channelSave));
% end

desiredSlices = (mov_size(3) / nChannelsTmp) / nVolumesTmp
nDiscardedExtra = nSlicesTmp - desiredSlices	
if desiredSlices ~= nSlicesTmp  % input (processed) tiff does not have discard removed, or has extra flyback frames removed.
   if nDiscardTmp == 0
       % This means discard frames were not specified and acquired, and flyback frames removed from top in processed tiff.
       extra_flyback_top = true;
       nDiscardTmp = nSlicesTmp - desiredSlices;
       false_discard = true;
   elseif nDiscardTmp > 0
       % Discard frames were specified/acquired but extra flyback frames removed from top of stack
       extra_flyback_top = true;
       false_discard = false;
   end
else
   extra_flyback_top = false;
   false_discard = false;
end
nSlicesSelected = desiredSlices; %nSlicesTmp - nDiscardTmp;

% Rewrite SI meta data for TIFF to match substack
% processing:
scanImageMetadata.SI.hStackManager.numSlices = nSlicesSelected;
scanImageMetadata.SI.hFastZ.numDiscardFlybackFrames = 0;
scanImageMetadata.SI.hFastZ.numFramesPerVolume = scanImageMetadata.SI.hStackManager.numSlices;
scanImageMetadata.SI.hStackManager.zs = scanImageMetadata.SI.hStackManager.zs(nDiscardedExtra+1:end);
scanImageMetadata.SI.hFastZ.discardFlybackFrames = 0;  % Need to disflag this so that parseScanimageTiff (from Acquisition2P) takes correct n slices
nFramesSelected = nChannelsTmp*nSlicesSelected*nVolumesTmp

metanames = fieldnames(scanImageMetadata);
for field=1:length(metanames)
   if strcmp(metanames{field}, 'SI')
       continue;
   else
        currfield = scanImageMetadata.(metanames{field});

        if extra_flyback_top && false_discard %falseDiscard
            % There are no additional empty flybacks at the end of volume, 
            % so just skip every nSlicesTmp, starting from corrected num of
            % nDiscard removed from top (i.e., nDiscardedExtra):
            startidxs = colon(nDiscardedExtra*nChannelsTmp+1, nChannelsTmp*(nSlicesTmp), length(currfield));
        elseif extra_flyback_top && ~false_discard
	    fprintf('TOP AND BOTTOM\n')
            % There were specified num of empty flybacks at end of volume, 
            % so remove those indices, if necessary, while also removing 
            % the removed frames at top:
            startidxs = colon(nDiscardedExtra*nChannelsTmp+1, nChannelsTmp*(nSlicesTmp+nDiscardTmp), length(currfield));
        else
            % There were empty flyybacks at end of volume, but correctly 
            % executed, s.t. no additional flybacks removed from top:
            startidxs = colon(1, nChannelsTmp*(nSlicesTmp+nDiscardTmp), length(currfield));
        end
        fprintf('N volumes based on start indices: %i\n', length(startidxs));

        if iscell(currfield)
           tmpfield = cell(1, nFramesSelected);
        else
           tmpfield = zeros(1, nFramesSelected);
        end
        newidx = 1;
        for startidx = startidxs
            %startidx
           tmpfield(newidx:newidx+(nSlicesSelected*nChannelsTmp - 1)) = currfield(startidx:startidx+(nSlicesSelected*nChannelsTmp - 1));
           newidx = newidx + (nSlicesSelected*nChannelsTmp);
        end
        scanImageMetadata.(metanames{field}) = tmpfield;
   end
end
%[movStruct, nSlices, nChannels] = parseProcessedScanimageTiff(mov, scanImageMetadata);

end
