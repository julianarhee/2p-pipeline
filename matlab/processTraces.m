function processTraces(D, winUnit, nWinUnits, varargin)

% Process traces:
% getTraces.m extracts raw traces using ROI masks (saves tracestruct)
% This function processes these raw traces and adds to tracestruct:
% 1. Remove bad frames.
% 2. Subtract rolling mean, and store DC offsets.
% -- winUnit :  The duration (sec) of a single unit (i.e., cycle or trial).
% 3. 

switch length(varargin)
    case 0
        trimEnd = false;
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

slicesToUse = D.slices;

for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    fprintf('Processing traces for Slice %02d...\n', currSlice);

    tracestruct = load(fullfile(D.tracesPath, D.traceNames{sidx}));
    for tiffIdx=1:nTiffs
        %tiffIdx
        meta = metastruct.file(tiffIdx);
        % If corrected traceMat not found, correct and save, otherwise
        % just load corrected tracemat:
        %if ~isfield(T, 'traceMat') || length(T.traceMat.file) < tiffIdx
            fprintf('TIFF %i: Creating new tracemat from processed traces...\n', tiffIdx);
            
            % 1. First, remove "bad" frames (too much motion), and
            % replace with Nans:
            currTraces = tracestruct.file(tiffIdx).rawTraces; % nxm mat, n=frames, m=rois
            
            tracestruct.file(tiffIdx).badFrames(tracestruct.file(tiffIdx).badFrames==tracestruct.file(tiffIdx).refframe) = []; % Ignore reference frame (corrcoef=1)

            bf = tracestruct.file(tiffIdx).badFrames;
            if length(bf) > 1
                assignNan = @(f) nan(size(f));
%                 tmpframes = arrayfun(@(i) assignNan(currTraces(:,i)), bf, 'UniformOutput', false);
                tmpframes = arrayfun(@(i) assignNan(currTraces(i,:)), bf, 'UniformOutput', false);
                currTraces(bf,:) = cat(1,tmpframes{1:end});
            end
            
            % 2.  Crop trailing frames (extra frames, not related to
            % stimulus), if needed:
            
            if trimEnd
                traces = currTraces(1:cropToFrame,:);
                %traces = currTraces(:, 1:cropToFrame);
            else
                traces = currTraces;
            end
            

            % 3. Next, get subtract rolling average from each trace
            % (each row of currTraces is the trace of an ROI).
            % Interpolate NaN frames if there are any.
            winsz = round(meta.si.siVolumeRate*winUnit*nWinUnits);
%             [traceMat, DCs] = arrayfun(@(i) subtractRollingMean(traces(i,:), winsz), 1:size(traces, 1), 'UniformOutput', false);
            [traceMat, DCs] = arrayfun(@(roi) subtractRollingMean(traces(:,roi), winsz), 1:size(traces,2), 'UniformOutput', false);
            traceMat = cat(2, traceMat{1:end});
            DCs = cat(2, DCs{1:end});
            traceMat = bsxfun(@plus, traceMat, DCs); % ROWs = tpoints, COLS = ROI
            
            tracestruct.file(tiffIdx).traceMat = traceMat;
            tracestruct.file(tiffIdx).winsz = winsz;
            tracestruct.file(tiffIdx).DCs = DCs;
            
            if trimEnd
               
                [untrimmedTraceMat, untrimmedDCs] = arrayfun(@(i) subtractRollingMean(currTraces(:,i), winsz), 1:size(currTraces,2), 'UniformOutput', false);
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