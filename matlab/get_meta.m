function [D, meta] = get_meta(D, dsoptions)

% --------------------------------------------------------------------
% Creates a meta struct that combines all the info for this expmt.
% --------------------------------------------------------------------
% - stimulation info (MW)
% - acquisition info (SI)

% Additionaly, adds parameters useful for creating database later...
% --------------------------------------------------------------------


metaInfo = dsoptions.meta;
nchannels = dsoptions.channels;

switch metaInfo
    case 'SI'

        siMetaName = sprintf('%s.mat', D.acquisitionName);

        %if ~exist(fullfile(D.sourceDir, siMetaName))
        if ~exist(fullfile(D.dataDir, siMetaName))

            fprintf('No motion-corrected META data found...\n');
            fprintf('Current acquisition is: %s\n', D.acquisitionName);
            fprintf('Parsing SI tiffs, and creating new meta file.\n');

            % Load and parse raw TIFFs and create SI-specific meta file:
            %movies = dir(fullfile(D.sourceDir,'*.tif'));
            movies = dir(fullfile(D.dataDir,'*.tif'));
            movies = {movies(:).name};
            %writeDir = fullfile(D.sourceDir, D.tiffSource);
            writeDir = fullfile(D.tiffSource);
            if ~exist(writeDir, 'dir')
                mkdir(writeDir)
            end
            switch D.preprocessing
                case 'fiji'
                    fprintf('No SI metadata for fiji tiffs.\nSelect reference:\n');
                    [fn, fpath, ~] = uigetfile();
                    sitmp = load(fullfile(fpath, fn));
                    siRefAcquisitionName = fieldnames(sitmp);
                    siRef = sitmp.(siRefAcquisitionName{1});
                    fprintf('Selected reference si metastruct from: %s.\n', siRef.acqName);
                    if length(movies)==1
                        siRefFidx = input('Select FILE idx from current reference acquisition:\n');
                        siRefMetaStruct = siRef.metaDataSI{siRefFidx};
                        %parseSIdata(D.acquisitionName, movies, D.dataDir, writeDir, siRefMetaStruct);
			parseSIdata(D, movies, writeDir, siRefMetaStruct);
                    else
                        siRefMetaStruct = siRef;
                        if length(movies) ~= length(siRef.metaDataSI)
                            siRefStartIdx = input('Select idx for FIRST file in reference meta stuct:\n');
                            if isempty(siRefStartIdx)
                                siRefStartIdx = 1;
                            end
                        else
                            siRefStartIdx = 1;
                        end
                        %parseSIdata(D.acquisitionName, movies, D.dataDir, writeDir, siRefMetaStruct, siRefStartIdx);
			parseSIdata(D, movies, writeDir, siRefMetaStruct, siRefStartIdx);

                    end

                otherwise
		    parseSIdata(D, movies, writeDir);
%                     if D.metaonly || D.processedtiffs
%                         %parseSIdata(D.acquisitionName, movies, D.dataDir, writeDir, [], [], D.metaonly, D.processedtiffs);
% 			parseSIdata(D, movies, writeDir);
% 
%                     else
%                         parseSIdata(D.acquisitionName, movies, D.dataDir, writeDir);
%                     end
            end

            % Load newly-created meta struct:
            %siMeta = load(fullfile(sourceDir, siMetaName));
            %meta = struct();
            %meta.(acquisitionName) = siMeta.metaDataSI; % Have info for each file, but this was not corrected in previosuly run MCs...
        end
        %
        % Sort Parsed files into separate directories if needed:
        tmpchannels = dir(D.tiffSource);
        tmpchannels = tmpchannels(arrayfun(@(x) ~strcmp(x.name(1),'.'), tmpchannels));
        tmpchannels = tmpchannels([tmpchannels.isdir]);
        tmpchannels = {tmpchannels(:).name}';
        %if length(dir(fullfile(D.sourceDir, D.tiffSource, tmpchannels{1}))) > length(tmpchannels)+2
        if isempty(tmpchannels) || any(strfind(D.tiffSource, 'Parsed'))
            sort_parsed_tiffs(D, nchannels);
        end

        % Creata META with SI (acquisition) and MW (experiment) info:
        meta = createMetaStruct(D);
        fprintf('Created meta struct for current acquisition,\n');

    case 'manual' 
        % No motion-correction/processing, just using raw TIFFs.
        nVolumes = 350;
        nSlices = 20;
        nDiscard = 0;

        nFramesPerVolume = nSlices + nDiscard;
        nTotalFrames = nFramesPerVolume * nVolumes;

        %.....
end

D.metaPath = meta.metaPath; 
D.nTiffs = meta.nTiffs 
D.nChannels = meta.nChannels; 
D.stimType = meta.stimType; 
 
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D'); 
 
end
