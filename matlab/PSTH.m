% PSTHs.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

acquisition_info;

%%

% -------------------------------------------------------------------------
% Parse FILES by SLICE:
% -------------------------------------------------------------------------
% For each slice:
% 1. Parse each ROI's trace into trials.
% 2. Store parsed trials for that slice as struct containing slice no, file
% no, stimulus ID, ROI no...
% 3. Repeat for each file.
% 4. Sort parsed trials by stimulus-type.

% Get meta info:
M = load(D.metaPath);
condTypes = M.condTypes;

% Get trace struct names:
traceNames = dir(fullfile(D.tracesPath, '*.mat'));
traceNames = {traceNames(:).name}';

slicesToUse = D.slices;

skipFirst = true;
ITI = 2;

trialStruct = struct();
for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    
%     M = load(maskPaths{sidx});
%     masks = M.masks;
%     clear M;
        
    T = load(fullfile(D.tracesPath, traceNames{sidx}));
    
    mwTrials = struct();
    mwEpochIdxs = struct();
    siEpochIdxs = struct();
    siTrials = struct();
    roiTraces = struct();
    roiTimes = struct();
    mwTimes = struct();
    deltaFs = struct();
    for stimIdx=1:length(condTypes)
        currStim = condTypes{stimIdx};
        mwTrials.(currStim) = [];
        mwEpochIdxs.(currStim) = [];
        siEpochIdxs.(currStim) = [];
        siTrials.(currStim) = [];
        roiTraces.(currStim) = [];
        roiTimes.(currStim) = [];
        mwTimes.(currStim) = [];
        deltaFs.(currStim) = [];
    end
    
    for fidx=1:nTiffs
        
        meta = M.file(fidx);
        volumeIdxs = currSlice:meta.si.nFramesPerVolume:meta.si.nTotalFrames;
        
        currTraces = T.traces.file{fidx};
        avgY = T.avgImage.file{fidx};
        
        currRun = meta.mw.runName;
        currMWcodes = meta.mw.pymat.(currRun).stimIDs;
        currTrialIdxs = meta.mw.pymat.(currRun).idxs + 1; % bec python 0-indexes
        
        currTiff = sprintf('tiff%i', fidx);
        for trialIdx=1:2:length(currMWcodes)
            if skipFirst && trialIdx==1
                continue;
            end
            if trialIdx < currTrialIdxs(end)
                currEnd = trialIdx + 2;
            else
                currEnd = length(meta.mw.mwSec);
            end
            currStim = sprintf('stim%i', currMWcodes(trialIdx));
            if ~isfield(mwTrials, currStim)
                mwTrials.(currStim).(currTiff){1} = meta.mw.mwSec(trialIdx-1:currEnd);
            else
                if ~isfield(mwTrials.(currStim), currTiff)
                    mwTrials.(currStim).(currTiff){1} = meta.mw.mwSec(trialIdx-1:currEnd);
                else
                    mwTrials.(currStim).(currTiff){end+1} = meta.mw.mwSec(trialIdx-1:currEnd);
                end
            end
            mwEpochIdxs.(currStim){end+1} = trialIdx-1:currEnd;
        end
    end %
    
%     siTrials = struct();
%     nTrials = struct();
%     roiTraces = struct();
%     roiTimes = struct();
%     mwTimes = struct();

    nTrials = struct();
    for stimIdx=1:length(condTypes) 

        currStim = condTypes{stimIdx};
        nTrials.(currStim) = 0;
%         roiTraces.(currStim) = [];
%         roiTimes.(currStim) = [];
%         mwTimes.(currStim) = [];
%         deltaFs.(currStim) = [];
        
        if ~isfield(mwTrials, currStim)
            fprintf('File %s not found for stim %s\n', currTiff, currStim);
            continue;
        end
            
        currTiffs = fieldnames(mwTrials.(currStim));
        for tiffIdx=1:length(currTiffs)
            %display(currTiffs{tiffIdx});
            currTrials = mwTrials.(currStim).(currTiffs{tiffIdx});
            for trialIdx=1:length(currTrials)
                currSecs = mwTrials.(currStim).(currTiffs{tiffIdx}){trialIdx};
                tPre = currSecs(1) + 1;
                tOn = currSecs(2);
                tOff = currSecs(3);
                if length(currSecs) < 4
                    continue;
                    fprintf('Not enough tstamps found in stim %s, file %i, trial %i.\n', currStim, tiffIdx, trialIdx);
                    tPost = tOff + ITI;
                else
                    tPost = currSecs(4);
                end
                sliceSecs = meta.mw.siSec(volumeIdxs);
                
                framePre = find(abs(sliceSecs-tPre) == min(abs(sliceSecs-tPre)));
                frameOn = find(abs(sliceSecs-tOn) == min(abs(sliceSecs-tOn)));
                frameOff = find(abs(sliceSecs-tOff) == min(abs(sliceSecs-tOff)));
                framePost = find(abs(sliceSecs-tPost) == min(abs(sliceSecs-tPost)));
                frameEpochIdxs = [framePre frameOn frameOff framePost];
                siTrials.(currStim).(currTiffs{tiffIdx}){trialIdx} = sliceSecs(frameEpochIdxs);
                
                nTrials.(currStim) = nTrials.(currStim) + 1;
                frameidxs = frameEpochIdxs(1):frameEpochIdxs(end);
%                 if length(frameidxs)>23
%                     frameidxs = frameidxs(1:23);
%                 end
                roiTraces.(currStim){end+1} = T.traces.file{tiffIdx}(:, frameidxs);
                roiTimes.(currStim){end+1} = sliceSecs(frameidxs) - tOn;
                %roiTimes.(currStim){end+1} = sliceSecs(frameidxs);
                mwTimes.(currStim){end+1} = currSecs;
                siEpochIdxs.(currStim){end+1} = frameidxs;
                
                % Calculate dFs:
                ctraces = T.traces.file{tiffIdx}(:, frameidxs);
                baselines = mean(T.traces.file{tiffIdx}(:,frameidxs(1):frameOn-1), 2);
                baselineMat = repmat(baselines, [1, size(ctraces,2)]);
                deltaFs.(currStim){end+1} = ((ctraces - baselineMat) ./ baselineMat)*100;
            end
        end
            
    end
    
    trialStruct.slice(currSlice).sliceIdx = currSlice;
    trialStruct.slice(currSlice).mwTrials = mwTrials;
    trialStruct.slice(currSlice).siTrials = siTrials;
    trialStruct.slice(currSlice).roiTraces = roiTraces;
    trialStruct.slice(currSlice).roiTimes = roiTimes;
    trialStruct.slice(currSlice).mwEpochIdxs = mwEpochIdxs;
    trialStruct.slice(currSlice).siEpochIdxs = siEpochIdxs;
    trialStruct.slice(currSlice).deltaFs = deltaFs;
    trialStruct.slice(currSlice).nTrials = nTrials;
end

%%

for sidx = slicesToUse
    currSlice = trialStruct.slice(sidx).sliceIdx;
    
    roi = 20;
    for stimIdx=1:length(condTypes)
        currStim = condTypes{stimIdx};
        
        currDfs = trialStruct.slice(sidx).deltaFs.(currStim);
        currTimes = trialStruct.slice(sidx).roiTimes.(currStim);
        for dfIdx=1:length(currDfs)
            t = currTimes{dfIdx}; % - currTimes{dfIdx}(1);
            df = currDfs{dfIdx}(roi,:);
            plot(t, df, 'k', 'LineWidth', 0.5);
            hold on;
        end
        trialsBySlice = cat(3, currDfs{:});
        meanTracesCurrSlice = mean(trialsBySlice, 3);
        plot(t, meanTracesCurrSlice(roi,:), 'k', 'LineWidth', 2);
        
        % ---
        for roi=1:size(meanTracesCurrSlice,1)
            plot(t, meanTracesCurrSlice(roi,:), 'k', 'LineWidth', .2);
            hold on;
        end
        % ---

        
        
%         for roi=1:size(meanTracesCurrSlice,1)
%             plot(meanTracesCurrSlice(roi,:));
%         end
        
    end
    
    y = T.traces.file{tiffIdx}(roi,:);
    dff = (y - mean(y))./mean(y);
    plot(meta.mw.siSec(volumeIdxs), dff);
    hold on;
    stimOnsets = meta.mw.mwSec(meta.mw.stimStarts);
    stimOffsets = meta.mw.mwSec(2:2:end);
    for onset=1:length(stimOnsets)
        %line([stimOnsets(onset) stimOnsets(onset)], get(gca, 'ylim'));
        ylims = get(gca, 'ylim');
        v = [stimOnsets(onset) ylims(1); stimOffsets(onset) ylims(1);...
            stimOffsets(onset) ylims(2); stimOnsets(onset) ylims(2)];
        f = [1 2 3 4];
        patch('Faces',f,'Vertices',v,'FaceColor','red', 'FaceAlpha', 0.2)
        hold on;
    end
    
    stimNames = fieldnames(mwTrials);
    for stimIdx=1:length(stimNames)
        fprintf('%i trials found for %s.\n', length(fieldnames(mwTrials.(stimNames{stimIdx}))), stimNames{stimIdx});
    end
    
    
    
    v = [0 0; 1 0; 1 1; 0 1];
f = [1 2 3 4];
patch('Faces',f,'Vertices',v,'FaceColor','red')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    siTrials = struct();
    for stimIdx=1:length(condTypes)
        if ~isfield(mwTrials, condTypes{stimIdx})
            fprintf('
        


            
            
            
    
    
    
end