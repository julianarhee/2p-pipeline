function process_traces_3Dnmf(D, winUnit, nWinUnits, varargin)

% Process traces:
% get_traces.m extracts raw traces using ROI masks (saves tracestruct)
% This function processes these raw traces and adds to tracestruct:
% 1. Remove bad frames.
% 2. Subtract rolling mean, and store DC offsets.
% -- winUnit :  The duration (sec) of a single unit (i.e., cycle or trial).
% 3. 

switch length(varargin)
    case 0
        trimEnd = false
    case 1
        trimEnd = true;
        cropToFrame = varargin{1};
    case 2
        trimEnd = true;
        cropToFrame = varargin{1};
end

metastruct = load(D.metaPath);
condTypes = metastruct.condTypes;
nTiffs = metastruct.nTiffs;

nslices = length(D.traceNames);

% DO the stupid 3D thing first:

for tiffIdx=1:nTiffs
    
    tracestruct = load(fullfile(D.tracesPath, D.traceNames3D{tiffIdx}));

    meta = metastruct.file(tiffIdx);
    
    % If corrected traceMat not found, correct and save, otherwise
    % just load corrected tracemat:
    %if ~isfield(T, 'traceMat') || length(T.traceMat.file) < tiffIdx
        fprintf('TIFF %i: Creating new tracemat from processed traces...\n', tiffIdx);

        % 1. First, remove "bad" frames (too much motion), and
        % replace with Nans:
        currTraces = tracestruct.rawTraces; % nxm mat, n=frames, m=rois
%         if isfield(tracestruct, 'inferredTraces')
%             currInferredTraces = tracestruct.inferredTraces;
%         end        
        if isfield(tracestruct, 'dfTracesNMF')
            currRawTracesNMF = tracestruct.rawTracesNMF;
            currDfTracesNMF = tracestruct.dfTracesNMF;
            currDetrendedNMF = tracestruct.detrendedNMF;
            inferred = true;
        else
            inferred = false;
        end
        
%         tracestruct.file(tiffIdx).badFrames(tracestruct.file(tiffIdx).badFrames==tracestruct.file(tiffIdx).refframe) = []; % Ignore reference frame (corrcoef=1)

%         bf = tracestruct.file(tiffIdx).badFrames;
%         if length(bf) > 1
%             assignNan = @(f) nan(size(f));
% %                 tmpframes = arrayfun(@(i) assignNan(currTraces(:,i)), bf, 'UniformOutput', false);
%             tmpframes = arrayfun(@(i) assignNan(currTraces(i,:)), bf, 'UniformOutput', false);
%             currTraces(bf,:) = cat(1,tmpframes{1:end});
%         end

        % 2.  Crop trailing frames (extra frames, not related to
        % stimulus), if needed:

        if trimEnd
            traces = currTraces(1:cropToFrame,:);
            if inferred
                rawTracesNMF = currRawTracesNMF(1:cropToFrame,:);
                dfTracesNMF = currDfTracesNMF(1:cropToFrame,:);
                detrendedNMF = currDetrendedNMF(1:cropToFrame,:);
            end
            %traces = currTraces(:, 1:cropToFrame);
        else
            traces = currTraces;
            if inferred
                rawTracesNMF = currRawTracesNMF;
                dfTracesNMF = currDfTracesNMF;
                detrendedNMF = currDetrendedNMF;
            end
        end


        % 3. Next, get subtract rolling average from each trace
        % (each row of currTraces is the trace of an ROI).
        % Interpolate NaN frames if there are any.
        winsz = round(meta.si.siVolumeRate*winUnit*nWinUnits);
        %[traceMat, DCs] = arrayfun(@(i) subtract_rolling_mean(traces(i,:), winsz), 1:size(traces, 1), 'UniformOutput', false);
        [traceMat, DCs] = arrayfun(@(roi) subtract_rolling_mean(traces(:,roi), winsz), 1:size(traces,2), 'UniformOutput', false);
        traceMat = cat(2, traceMat{1:end});
        DCs = cat(2, DCs{1:end});
        traceMatDC = bsxfun(@plus, DCs, traceMat); % ROWs = tpoints, COLS = ROI
        
        tracestruct.traceMatDC = traceMatDC;
        tracestruct.traceMat = traceMat;
        tracestruct.winsz = winsz;
        tracestruct.DCs = DCs;
        if inferred 
            %tracestruct.inferredTraceMat = inferredTraces;
            [rawTraceMatNMF, rawDCsNMF] = arrayfun(@(roi) subtract_rolling_mean(rawTracesNMF(:,roi), winsz), 1:size(rawTracesNMF,2), 'UniformOutput', false);
            rawTraceMatNMF = cat(2, rawTraceMatNMF{1:end});
            rawDCsNMF = cat(2, rawDCsNMF{1:end});
            rawTraceMatDCNMF = bsxfun(@plus, rawDCsNMF, rawTraceMatNMF); % ROWs = tpoints, COLS = ROI
            
            tracestruct.rawTraceMatDCNMF = rawTraceMatDCNMF;
            tracestruct.rawTraceMatNMF = rawTraceMatNMF;
            tracestruct.rawDCsNMF = rawDCsNMF;
            tracestruct.dfTraceMatNMF = dfTracesNMF;
            tracestruct.detrendedTraceMatNMF = detrendedNMF;
        end
        
        if trimEnd

            [untrimmedTraceMat, untrimmedDCs] = arrayfun(@(i) subtract_rolling_mean(currTraces(:,i), winsz), 1:size(currTraces,2), 'UniformOutput', false);
            untrimmedTraceMat = cat(2, untrimmedTraceMat{1:end});
            untrimmedDCs = cat(2, untrimmedDCs{1:end});
            untrimmedTraceMat = bsxfun(@plus, untrimmedDCs, untrimmedTraceMat);
            tracestruct.untrimmedTracemat = untrimmedTraceMat;
            tracestruct.untrimmedDCs = untrimmedDCs;
            tracestruct.cropToFrame = cropToFrame;
        end

        save(fullfile(D.tracesPath, D.traceNames3D{tiffIdx}), '-append', '-struct', 'tracestruct');
%         else
%             traceMat = T.traceMat.file{tiffIdx};
    %end

end



for sidx=1:nslices
    
tracestruct = load(fullfile(D.tracesPath, D.traceNames{sidx}));

for tiffIdx=1:nTiffs

    meta = metastruct.file(tiffIdx);
    
    % If corrected traceMat not found, correct and save, otherwise
    % just load corrected tracemat:
    %if ~isfield(T, 'traceMat') || length(T.traceMat.file) < tiffIdx
        fprintf('TIFF %i: Creating new tracemat from processed traces...\n', tiffIdx);

        % 1. First, remove "bad" frames (too much motion), and
        % replace with Nans:
         currTraces = tracestruct.file(tiffIdx).rawTraces; % nxm mat, n=frames, m=rois
%         if isfield(tracestruct, 'inferredTraces')
%             currInferredTraces = tracestruct.file(tiffIdx).inferredTraces;
%         end
        if isfield(tracestruct.file(tiffIdx), 'dfTracesNMF')
            currRawTracesNMF = tracestruct.file(tiffIdx).rawTracesNMF;
            currDfTracesNMF = tracestruct.file(tiffIdx).dfTracesNMF;
            currDetrendedNMF = tracestruct.file(tiffIdx).detrendedTracesNMF;
            inferred = true;
        else
            inferred = false;
        end
%         tracestruct.file(tiffIdx).badFrames(tracestruct.file(tiffIdx).badFrames==tracestruct.file(tiffIdx).refframe) = []; % Ignore reference frame (corrcoef=1)

%         bf = tracestruct.file(tiffIdx).badFrames;
%         if length(bf) > 1
%             assignNan = @(f) nan(size(f));
% %                 tmpframes = arrayfun(@(i) assignNan(currTraces(:,i)), bf, 'UniformOutput', false);
%             tmpframes = arrayfun(@(i) assignNan(currTraces(i,:)), bf, 'UniformOutput', false);
%             currTraces(bf,:) = cat(1,tmpframes{1:end});
%         end

        % 2.  Crop trailing frames (extra frames, not related to
        % stimulus), if needed:

        if trimEnd
            traces = currTraces(1:cropToFrame,:);
%             if isfield(tracestruct, 'inferredTraces')
%                 inferredTraces = currInferredTraces(1:cropToFrame,:);
%             end
            if inferred
                rawTracesNMF = currRawTracesNMF(1:cropToFrame,:);
                dfTracesNMF = currDfTracesNMF(1:cropToFrame,:);
                detrendedNMF = currDetrendedNMF(1:cropToFrame,:);
            end            
            %traces = currTraces(:, 1:cropToFrame);
        else
            traces = currTraces;
%             if isfield(tracestruct, 'inferredTraces')
%                 inferredTraces = currInferredTraces;
%             end
            if inferred
                rawTracesNMF = currRawTracesNMF; 
                dfTracesNMF = currDfTracesNMF;
                detrendedNMF = currDetrendedNMF;
            end
        end


        % 3. Next, get subtract rolling average from each trace
        % (each row of currTraces is the trace of an ROI).
        % Interpolate NaN frames if there are any.
        winsz = round(meta.si.siVolumeRate*winUnit*nWinUnits);
%             [traceMat, DCs] = arrayfun(@(i) subtract_rolling_mean(traces(i,:), winsz), 1:size(traces, 1), 'UniformOutput', false);
        [traceMat, DCs] = arrayfun(@(roi) subtract_rolling_mean(traces(:,roi), winsz), 1:size(traces,2), 'UniformOutput', false);
        traceMat = cat(2, traceMat{1:end});
        DCs = cat(2, DCs{1:end});
        traceMatDC = bsxfun(@plus, traceMat, DCs); % ROWs = tpoints, COLS = ROI

        tracestruct.file(tiffIdx).traceMatDC = traceMatDC;
        tracestruct.file(tiffIdx).traceMat = traceMat;
        tracestruct.file(tiffIdx).winsz = winsz;
        tracestruct.file(tiffIdx).DCs = DCs;
        
%         if isfield(tracestruct, 'inferredTraces')
%             tracestruct.file(tiffIdx).inferredTraceMat = inferredTraces;
%         end
        if inferred
            %tracestruct.inferredTraceMat = inferredTraces;
            [rawTraceMatNMF, rawDCsNMF] = arrayfun(@(roi) subtract_rolling_mean(rawTracesNMF(:,roi), winsz), 1:size(rawTracesNMF,2), 'UniformOutput', false);
            rawTraceMatNMF = cat(2, rawTraceMatNMF{1:end});
            rawDCsNMF = cat(2, rawDCsNMF{1:end});
            rawTraceMatDCNMF = bsxfun(@plus, rawTraceMatNMF, rawDCsNMF); % ROWs = tpoints, COLS = ROI

            tracestruct.file(tiffIdx).rawTraceMatDCNMF = rawTraceMatDCNMF;
            tracestruct.file(tiffIdx).rawTraceMatNMF = rawTraceMatNMF;
            tracestruct.file(tiffIdx).rawDCsNMF = rawDCsNMF;
            tracestruct.file(tiffIdx).dfTraceMatNMF = dfTracesNMF;
            tracestruct.file(tiffIdx).detrendedTraceMatNMF = detrendedNMF;
        end
        
        if trimEnd

            [untrimmedTraceMat, untrimmedDCs] = arrayfun(@(i) subtract_rolling_mean(currTraces(:,i), winsz), 1:size(currTraces,2), 'UniformOutput', false);
            untrimmedTraceMat = cat(2, untrimmedTraceMat{1:end});
            untrimmedDCs = cat(2, untrimmedDCs{1:end});
            untrimmedTraceMat = bsxfun(@plus, untrimmedDCs, untrimmedTraceMat);
            tracestruct.file(tiffIdx).untrimmedTracemat = untrimmedTraceMat;
            tracestruct.file(tiffIdx).untrimmedDCs = untrimmedDCs;
            tracestruct.file(tiffIdx).cropToFrame = cropToFrame;
        end

        save(fullfile(D.tracesPath, D.traceNames{sidx}), '-append', '-struct', 'tracestruct');
%         else
%             traceMat = T.traceMat.file{tiffIdx};
    %end

end

end
            
end
