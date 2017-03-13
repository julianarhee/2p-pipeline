function parseSIdata(acquisition_name, movies, sourceDir, writeDir)
% Parse and save TIFFs that are not preprocessed (motion corrected).
% Rename by Channel, Slice, and File.

%% Set params:

namingFunction = @defaultNamingFunction;

%% Load movies and motion correct
%Calculate Number of movies and arrange processing order so that
%reference is first
nMovies = length(movies);
movieOrder = 1:nMovies;

%Load movies one at a time in order, apply correction, and save as
%split files (slice and channel)
metaDataSI = {};

for movNum = movieOrder
    fprintf('\nLoading Movie #%03.0f of #%03.0f\n',movNum,nMovies),
    [mov, scanImageMetadata] = tiffRead(fullfile(sourceDir, movies{movNum}));

    fprintf('Mov size is: %s\n.', mat2str(size(mov)));
    fprintf('Mov type is: %s\n.', class(mov)); 
%     if obj.binFactor > 1
%         mov = binSpatial(mov, obj.binFactor);
%     end
%     
%     % Apply line shift:
%     fprintf('Line Shift Correcting Movie #%03.0f of #%03.0f\n', movNum, nMovies),
%     mov = correctLineShift(mov);
    try
        [movStruct, nSlices, nChannels] = parseScanimageTiff(mov, scanImageMetadata);
    catch
        error('parseScanimageTiff failed to parse metadata'),
    end
    clear mov
    
%     % Find motion:
%     fprintf('Identifying Motion Correction for Movie #%03.0f of #%03.0f\n', movNum, nMovies),
%     obj.motionCorrectionFunction(obj, movStruct, scanImageMetadata, movNum, 'identify');
%     
%     % Apply motion correction and write separate file for each
%     % slice\channel:
%     fprintf('Applying Motion Correction for Movie #%03.0f of #%03.0f\n', movNum, nMovies),
%     movStruct = obj.motionCorrectionFunction(obj, movStruct, scanImageMetadata, movNum, 'apply');
    
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
    
    metaDataSI{movNum} = scanImageMetadata;

end

%metaDataSI_fn = sprintf('%s_meta', acquisition_name);

save(fullfile(sourceDir, acquisition_name), 'metaDataSI')
display('Parsing Completed! Metadata saved to:')
display(fullfile(sourceDir, acquisition_name));

end


function movFileName = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

movFileName = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end
