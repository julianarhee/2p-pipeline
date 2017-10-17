function tracestruct_names = get_processed_traces(I, A, win_unit, num_units, varargin)

% Process traces:
% get_traces.m extracts raw traces using ROI masks (saves tracestruct)
% This function processes these raw traces and adds to tracestruct:
% 1. Remove bad frames.
% 2. Subtract rolling mean, and store DC offsets.
% -- win_unit :  The duration (sec) of a single unit (i.e., cycle or trial).
% 3. 

tracestruct_names = {};

switch length(varargin)
    case 0
        trim_end = false;
    case 1
        trim_end = true;
        last_frame = varargin{1};
    case 2
        trim_end = true;
        last_frame = varargin{1};
end

funcdir_idx = find(arrayfun(@(c) any(strfind(A.simeta_path{c}, I.functional)), 1:length(A.simeta_path))); 


simeta = load(A.simeta_path{funcdir_idx});
metastruct = simeta.SI;

for sidx = 1:length(I.slices)
    sl = I.slices(sidx);
    fprintf('Processing traces for Slice %02d...\n', sl);
    
    tracestruct_name = sprintf('traces_Slice%02d_Channel%02d.mat', sl, I.signal_channel);
    tracestruct = load(fullfile(A.trace_dir, I.roi_id, tracestruct_name));
    ntiffs = length(tracestruct.file);
    for fidx=1:ntiffs
        if length(metastruct.file)==1
            meta = metastruct.file
        else
            meta = metastruct.file(fidx);
        end

        fprintf('TIFF %i: Creating new tracemat from processed traces...\n', fidx);
        
        rawtracemat = tracestruct.file(fidx).rawtracemat;
        
        % TODO:  use MC-evaluation to remove/replace "bad" frames:
%         % 1. First, remove "bad" frames (too much motion), and
%         % replace with Nans:
%         currTraces = tracestruct.file(fidx).rawTraces; % nxm mat, n=frames, m=rois
% 
%         if isfield(tracestruct.file(fidx), 'badFrames')
%             tracestruct.file(fidx).badFrames(tracestruct.file(fidx).badFrames==tracestruct.file(fidx).refframe) = []; % Ignore reference frame (corrcoef=1)
% 
%             bf = tracestruct.file(fidx).badFrames;
%             if length(bf) > 1
%                 assignNan = @(f) nan(size(f));
% %                 tmpframes = arrayfun(@(i) assignNan(currTraces(:,i)), bf, 'UniformOutput', false);
%                 tmpframes = arrayfun(@(i) assignNan(currTraces(i,:)), bf, 'UniformOutput', false);
%                 currTraces(bf,:) = cat(1,tmpframes{1:end});
%             end
% 
%         end

        % 2.  Crop trailing frames (extra frames, not related to
        % stimulus), if needed:

        if trim_end && size(rawtracemat,1)>last_frame 
            traces = rawtracemat(1:last_frame,:);
        else
            traces = rawtracemat;
        end


        % 3. Next, get subtract rolling average from each trace
        % (each row of rawtracemat is the trace of an ROI).
        % Interpolate NaN frames if there are any.
        if ischar(meta.siVolumeRate)
            winsz = floor(str2num(meta.siVolumeRate)*win_unit*num_units);
        else
            winsz = floor(meta.siVolumeRate*win_unit*num_units);
        end
        [tracemat, DCs] = arrayfun(@(roi) subtract_rolling_mean(traces(:,roi), winsz), 1:size(traces,2), 'UniformOutput', false);
        tracemat = cat(2, tracemat{1:end});
        DCs = cat(2, DCs{1:end});
        tracematDC = bsxfun(@plus, tracemat, DCs); % ROWs = tpoints, COLS = ROI

        tracestruct.file(fidx).tracematDC = tracematDC;
        tracestruct.file(fidx).tracemat = tracemat;
        tracestruct.file(fidx).winsz = winsz;
        tracestruct.file(fidx).DCs = DCs;
            
        % 4. Also save full trace (not trimmed):
        if trim_end
            [untrimmedtracemat, untrimmedDCs] = arrayfun(@(i) subtract_rolling_mean(rawtracemat(:,i), winsz), 1:size(rawtracemat,2), 'UniformOutput', false);
            untrimmedtracemat = cat(2, untrimmedtracemat{1:end});
            untrimmedDCs = cat(2, untrimmedDCs{1:end});
            untrimmedtracemat = bsxfun(@plus, untrimmedDCs, untrimmedtracemat);
            tracestruct.file(fidx).untrimmedtracemat = untrimmedtracemat;
            tracestruct.file(fidx).untrimmedDCs = untrimmedDCs;
            tracestruct.file(fidx).last_frame = last_frame;
        end
        
    end
    
    % Save slice ROIs (including all files):
    save(fullfile(A.trace_dir, I.roi_id, tracestruct_name), '-append', '-struct', 'tracestruct');
    tracestruct_names{end+1} = tracestruct_name;

end
            
end
