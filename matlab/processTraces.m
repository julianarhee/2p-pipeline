function processTraces(D, winUnit, varargin)

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

M = load(D.metaPath);
condTypes = M.condTypes;
nTiffs = M.nTiffs;

slicesToUse = D.slices;

for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);

    T = load(fullfile(D.tracesPath, D.traceNames{sidx}));
    for tiffIdx=1:nTiffs
        meta = M.file(tiffIdx);
        % If corrected traceMat not found, correct and save, otherwise
        % just load corrected tracemat:
        %if ~isfield(T, 'traceMat') || length(T.traceMat.file) < tiffIdx

            % 1. First, remove "bad" frames (too much motion), and
            % replace with Nans:
            currTraces = T.rawTraces.file{tiffIdx};
            T.badFrames.file{tiffIdx}(T.badFrames.file{tiffIdx}==T.refframe.file(tiffIdx)) = []; % Ignore reference frame (corrcoef=1)

            bf = T.badFrames.file{tiffIdx};
            if length(bf) > 1
                assignNan = @(f) nan(size(f));
                tmpframes = arrayfun(@(i) assignNan(currTraces(:,i)), bf, 'UniformOutput', false);
                currTraces(:,bf) = cat(2,tmpframes{1:end});
            end
            
            % 2.  Crop trailing frames (extra frames, not related to
            % stimulus), if needed:
            if trimEnd
                traces = currTraces(:, 1:cropToFrame);
            else
                traces = currTraces;
            end
            

            % 3. Next, get subtract rolling average from each trace
            % (each row of currTraces is the trace of an ROI).
            % Interpolate NaN frames if there are any.
            winsz = round(meta.si.siVolumeRate*winUnit*2);
            [traceMat, DCs] = arrayfun(@(i) subtractRollingMean(traces(i, :), winsz), 1:size(traces, 1), 'UniformOutput', false);
            traceMat = cat(1, traceMat{1:end});
            DCs = cat(1, DCs{1:end});
            traceMat = bsxfun(@plus, DCs, traceMat);
            
            T.traceMat.file{tiffIdx} = traceMat;
            T.winsz.file(tiffIdx) = winsz;
            T.DCs.file{tiffIdx} = DCs;
            
            if trimEnd
                [untrimmedTraceMat, untrimmedDCs] = arrayfun(@(i) subtractRollingMean(currTraces(i, :), winsz), 1:size(currTraces, 1), 'UniformOutput', false);
                untrimmedTraceMat = cat(1, untrimmedTraceMat{1:end});
                untrimmedDCs = cat(1, untrimmedDCs{1:end});
                untrimmedTraceMat = bsxfun(@plus, untrimmedDCs, untrimmedTraceMat);
                T.untrimmedTracemat.file{tiffIdx} = untrimmedTraceMat;
                T.untrimmedDCs.file{tiffIdx} = untrimmedDCs;
                T.cropToFrame.file{tiffIdx} = cropToFrame;
            end

            save(fullfile(D.tracesPath, D.traceNames{sidx}), '-append', '-struct', 'T');
%         else
%             traceMat = T.traceMat.file{tiffIdx};
        %end

    end
end
            
end