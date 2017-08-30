function parseSIdata(D, movies, writeDir, varargin)
%function parseSIdata(acquisition_name, movies, sourceDir, writeDir, varargin)
% Parse and save TIFFs that are not preprocessed (motion corrected).
% Rename by Channel, Slice, and File.

%% Set params:

namingFunction = @defaultNamingFunction;
sourceDir = D.dataDir;
acquisition_name = D.acquisitionName;
processedtiffs = D.processedtiffs;
metaonly = D.metaonly;
corrected = D.corrected;

switch length(varargin)
    case 0
        refmeta = false;
        %metaonly = false;
	[parentDir, processedFolder, ~] = fileparts(sourceDir);
    case 1
        sirefs = varargin{1};
        sirefFidx = 1;
        refmeta = true;
	%metaonly = false;
    case 2
        sirefs = varargin{1};
        sirefFidx = varargin{2};
        refmeta = true;
        %metaonly = false;
    case 3
        refmeta = false;
        parentDir = varargin{3};

end
%% Load movies and motion correct
%Calculate Number of movies and arrange processing order so that
%reference is first
nMovies = length(movies);
movieOrder = 1:nMovies;

%Load movies one at a time in order, apply correction, and save as
%split files (slice and channel)
metaDataSI = {};

for movNum = movieOrder
    fprintf('\nLoading Movie #%03.0f of #%03.0f\n',movNum,nMovies)
    if refmeta
        refMovIdx = movNum + sirefFidx - 1; 
        scanImageMetadata = sirefs.metaDataSI{refMovIdx};
        [mov, ~] = tiffRead(fullfile(sourceDir, movies{movNum}));
    else
        if metaonly && ~processedtiffs
            fprintf('Only getting metadata...\n');
            scanImageMetadata = tiffReadMeta(fullfile(sourceDir, movies{movNum}));
        elseif metaonly && processedtiffs
            % meta source info comes from original tiffs:
            origMovies = dir(fullfile(parentDir,'*.tif'));
            origMovies = {origMovies(:).name};
            [~, scanImageMetadata] = tiffReadMeta(fullfile(parentDir, origMovies{movNum}));
            % but movie file comes from processed tiff:
            [mov, ~] = tiffRead(fullfile(sourceDir, movies{movNum}));
        else
            [mov, scanImageMetadata] = tiffRead(fullfile(sourceDir, movies{movNum}));
        end
    end
    
    % If using processedtiffs, check to make sure FOV isn't cropped/changed. Fix, if so:    
    if ~metaonly || processedtiffs
        fprintf('Mov size is: %s\n.', mat2str(size(mov)));
        fprintf('Mov type is: %s\n.', class(mov)); 
        fprintf('Pixels Per Line: %i\n', scanImageMetadata.SI.hRoiManager.pixelsPerLine)
        fprintf('Lines Per Frame: %i\n', scanImageMetadata.SI.hRoiManager.linesPerFrame)
        if size(mov,2)~=scanImageMetadata.SI.hRoiManager.pixelsPerLine
            fprintf('Cropping pixels per line to %i\n', size(mov,2));
            scanImageMetadata.SI.hRoiManager.pixelsPerLine = size(mov,2);
        end
        if size(mov,1)~=scanImageMetadata.SI.hRoiManager.linesPerFrame
            fprintf('Cropping lines per frame to %i\n', size(mov,1));
            scanImageMetadata.SI.hRoiManager.linesPerFrame = size(mov,1);
        end
    end
    
%     if obj.binFactor > 1
%         mov = binSpatial(mov, obj.binFactor);
%     end
%     
%     % Apply line shift:
%     fprintf('Line Shift Correcting Movie #%03.0f of #%03.0f\n', movNum, nMovies),
%     mov = correctLineShift(mov);

    if metaonly && ~processedtiffs
        fprintf('No movie read. Only getting metadata...\n');
        siStruct = scanImageMetadata.SI;
        fZ              = siStruct.hFastZ.enable; %siStruct.fastZEnable;
        nChannels = numel(siStruct.hChannels.channelSave); %numel(siStruct.channelsSave);
        nSlices = siStruct.hStackManager.numSlices + (fZ*siStruct.hFastZ.numDiscardFlybackFrames); % Slices are acquired at different locations (e.g. depths).
        discard = (fZ*siStruct.hFastZ.numDiscardFlybackFrames);
        nSlices = nSlices - discard; %nSlices-(fZ*siStruct.fastZDiscardFlybackFrames);
    else
        try
           if processedtiffs
               fprintf('Parsing processed SI tiff and getting adjusted meta data...\n');
               fprintf('Size of movie: %s\n', mat2str(size(mov)));
               nSlicesTmp = scanImageMetadata.SI.hStackManager.numSlices;
               nDiscardTmp = scanImageMetadata.SI.hFastZ.numDiscardFlybackFrames;
               nVolumesTmp = scanImageMetadata.SI.hFastZ.numVolumes;
               nChannelsTmp = numel(scanImageMetadata.SI.hChannels.channelSave);
   	       desiredSlices = (size(mov, 3) / nChannelsTmp) / nVolumesTmp
	       nDiscardedExtra = nSlicesTmp - desiredSlices;	
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

%                expectedSlices = (size(mov, 3) / nChannelsTmp) / nVolumesTmp; 
%                 if expectedSlices ~= (nSlicesTmp - nDiscardTmp)
%                     nDiscardTmp = nSlicesTmp - expectedSlices;
%                     falseDiscard = true
%                 else
%                     falseDiscard = false
%                 end
               nSlicesSelected = desiredSlices; %nSlicesTmp - nDiscardTmp;
	       
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
		   	   % there are no additional empty flybacks at the end of volume, so just skip every nSlicesTmp, starting from corrected num nDiscard removed from top:
			   startidxs = colon(nDiscardTmp*nChannelsTmp+1, nChannelsTmp*(nSlicesTmp), length(currfield));
			   fprintf('N volumes based on start indices: %i\n', length(startidxs));
		       elseif extra_flyback_top && ~false_discard
			   % There were specified num of empty flybacks at end of volume, so remove those indices, if necessary, while also removing the removed frames at top:
			   startidxs = colon(nDiscardTmp*nChannelsTmp+1, nChannelsTmp*(nSlicesTmp+nDiscardTmp), length(currfield));
		       else
			   % There were empty flyybacks at end of volume, but correctly executed, s.t. no additional flybacks removed from top:
			   startidxs = colon(1:nChannelsTmp*(nSlicesTmp+nDiscardTmp), length(currfield));
		       end
                       if iscell(currfield)
                           tmpfield = cell(1, nFramesSelected);
                       else
                           tmpfield = zeros(1, nFramesSelected);
                       end
                       newidx = 1;
                       for startidx = startidxs
                           tmpfield(newidx:newidx+(nSlicesSelected*nChannelsTmp - 1)) = currfield(startidx:startidx+(nSlicesSelected*nChannelsTmp - 1));
                           newidx = newidx + (nSlicesSelected*nChannelsTmp);
                       end
                       scanImageMetadata.(metanames{field}) = tmpfield;
                   end
               end
               %[movStruct, nSlices, nChannels] = parseProcessedScanimageTiff(mov, scanImageMetadata);
           end
           
           [movStruct, nSlices, nChannels] = parseScanimageTiff(mov, scanImageMetadata);
           
        catch
            error('parseScanimageTiff failed to parse metadata'),
        end
        clear mov
    end
    
%     % Find motion:
%     fprintf('Identifying Motion Correction for Movie #%03.0f of #%03.0f\n', movNum, nMovies),
%     obj.motionCorrectionFunction(obj, movStruct, scanImageMetadata, movNum, 'identify');
%     
%     % Apply motion correction and write separate file for each
%     % slice\channel:
%     fprintf('Applying Motion Correction for Movie #%03.0f of #%03.0f\n', movNum, nMovies),
%     movStruct = obj.motionCorrectionFunction(obj, movStruct, scanImageMetadata, movNum, 'apply');
    
    if (~metaonly || processedtiffs) && ~corrected
        fprintf('Writing parsed tiffs to file...\n');
        for nSlice = 1:nSlices
            for nChannel = 1:nChannels
                % Create movie fileName and save in acq object
                movFileName = feval(namingFunction,acquisition_name, nSlice, nChannel, movNum);
    %             obj.correctedMovies.slice(nSlice).channel(nChannel).fileName{movNum} = fullfile(writeDir,movFileName);

    %             % Determine 3D-size of movie and store w/ fileNames
    %             obj.correctedMovies.slice(nSlice).channel(nChannel).size(movNum,:) = ...
    %                 size(movStruct.slice(nSlice).channel(nChannel).mov);            
                % Write corrected movie to disk
                fprintf('Writing Movie #%03.0f of #%03.0f\n',movNum,nMovies),
                try
                    tiffWrite(movStruct.slice(nSlice).channel(nChannel).mov, movFileName, writeDir) %, 'int16');
                    %tiffWrite(movStruct.slice(nSlice).channel(nChannel).mov, movFileName, writeDir, 'uint16');
                catch
                    % Sometimes, disk access fails due to intermittent
                    % network problem. In that case, wait and re-try once:
                    pause(60);
                    tiffWrite(movStruct.slice(nSlice).channel(nChannel).mov, movFileName, writeDir) %, 'int16');
                    %tiffWrite(movStruct.slice(nSlice).channel(nChannel).mov, movFileName, writeDir, 'uint16');
                end
            end
        end
    end
    
    metaDataSI{movNum} = scanImageMetadata;

end

%metaDataSI_fn = sprintf('%s_meta', acquisition_name);
eval([acquisition_name '= struct();'])
eval([(acquisition_name) '.metaDataSI = metaDataSI'])
mfilename = sprintf('%s', acquisition_name);

save(fullfile(sourceDir, acquisition_name), mfilename)
display('Parsing Completed! Metadata saved to:')
display(fullfile(sourceDir, acquisition_name));

end


function movFileName = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

movFileName = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end
