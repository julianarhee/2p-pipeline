% PSTHs.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

acquisition_info;

%

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
    siTraces = struct();
    siTimes = struct();
    mwTimes = struct();
    deltaFs = struct();
    for stimIdx=1:length(condTypes)
        currStim = condTypes{stimIdx};
        mwTrials.(currStim) = [];
        mwEpochIdxs.(currStim) = [];
        siEpochIdxs.(currStim) = [];
        siTrials.(currStim) = [];
        siTraces.(currStim) = [];
        siTimes.(currStim) = [];
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
            
            currTraces = T.traces.file{tiffIdx};
            winsz = round(meta.si.siVolumeRate*3*2);
            traceMat = arrayfun(@(i) subtractRollingMean(currTraces(i, :), winsz), 1:size(currTraces, 1), 'UniformOutput', false);
            traceMat = cat(1, traceMat{1:end});
            
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
                %end
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
                    siTraces.(currStim){end+1} = traceMat(:, frameidxs);
                    siTimes.(currStim){end+1} = sliceSecs(frameidxs) - tOn;
                    %roiTimes.(currStim){end+1} = sliceSecs(frameidxs);
                    mwTimes.(currStim){end+1} = [tPre tOn tOff tPost] - tOn; %currSecs - tOn;
                    siEpochIdxs.(currStim){end+1} = frameEpochIdxs;
                    %siOnsetTime.(currStim){end+1} = sliceSecs(tOn)

                    % Calculate dFs:
                    ctraces = traceMat(:, frameidxs);
                    baselines = mean(traceMat(:,frameidxs(1):frameOn-1), 2);
                    baselineMat = repmat(baselines, [1, size(ctraces,2)]);
                    deltaFs.(currStim){end+1} = ((ctraces - baselineMat) ./ baselineMat)*100;
                end
            end
        end
            
    end
    
    trialStruct.slice(currSlice).sliceIdx = currSlice;
    trialStruct.slice(currSlice).mwTrials = mwTrials;
    trialStruct.slice(currSlice).siTrials = siTrials;
    trialStruct.slice(currSlice).siTraces = siTraces;
    trialStruct.slice(currSlice).siTimes = siTimes;
    trialStruct.slice(currSlice).mwTimes = mwTimes;
    trialStruct.slice(currSlice).mwEpochIdxs = mwEpochIdxs;
    trialStruct.slice(currSlice).siEpochIdxs = siEpochIdxs;
    trialStruct.slice(currSlice).deltaFs = deltaFs;
    trialStruct.slice(currSlice).nTrials = nTrials;
end

trialStructName = 'stimReps.mat';
save_struct(outputDir, trialStructName, trialStruct);


%%

currRoi = 50;
currSlice = 10;

iTraces = struct();

figure();
pRows = 2;

mwTimes = trialStruct.slice(currSlice).mwTimes;
siTimes = trialStruct.slice(currSlice).siTimes;
siTraces = trialStruct.slice(currSlice).siTraces;

%deltaFs = trialStruct.slice(currSlice).deltaFs;


nStim = length(fieldnames(siTraces));
pIdx = 1;
for sidx=1:length(fieldnames(siTraces))
    currStim = sprintf('stim%i', sidx);
    iTraces.(currStim) = [];

    tMW = cat(1, mwTimes.(currStim){1:end});
    currnpoints = [];
    for tsi=1:length(siTimes.(currStim))
        currnpoints = [currnpoints length(siTimes.(currStim){tsi})];
    end
    nFrames = median(currnpoints);
    
    interpfunc = @(x) linspace(x(1), x(end), nFrames);
    mwinterpMat = cell2mat(arrayfun(@(i) interpfunc(tMW(i,:)), 1:size(tMW,1), 'UniformOutput', false)');
    %nFrames = size(mwinterpMat,2);

    nTrials= size(tMW,1);
    for trial=1:nTrials
        nRois = size(siTraces.(currStim){trial},1);
        roiInterp = zeros(nRois, nFrames);
        for roi=1:nRois
            F = griddedInterpolant(siTimes.(currStim){trial}, siTraces.(currStim){trial}(roi,:));
            roiInterp(roi,:) = F(mwinterpMat(trial,:));    
        end
        iTraces.(currStim){end+1} = roiInterp;    
    end
    interpTraceMat = cat(1, iTraces.(currStim){1:end});
    currRoiTracesInterp = interpTraceMat(currRoi:nRois:end, :);
    
    [baseR, baseC] = find(mwinterpMat>=0);
    baselineMatTmp = currRoiTracesInterp.*(mwinterpMat<0);
    baselineMatTmp(baseR, baseC) = NaN;
    baselines = nanmean(baselineMatTmp,2);
    baselineMat = repmat(baselines, [1, size(currRoiTracesInterp,2)]);
    dfMat = ((currRoiTracesInterp - baselineMat) ./ baselineMat)*100;
%     currnpoints = [];
%     for tsi=1:length(siTimes.(currStim))
%         currnpoints = [currnpoints length(siTimes.(currStim){tsi})];
%     end
%     minpoints = min(currnpoints);
%     if length(unique(currnpoints)) > 1
%         minpoints = min(currnpoints);
%         for tsi=1:length(siTimes.(currStim))
%             oIdx = find(siTimes.(currStim){tsi}>0);
%             if oIdx(1)==6
%                 siTimes.(currStim){tsi} = siTimes.(currStim){tsi}(1:minpoints);
%                 siTraces.(currStim){tsi} = siTraces.(currStim){tsi}(:, 1:minpoints);
%             elseif oIdx(1)==7
%                 siTimes.(currStim){tsi} = siTimes.(currStim){tsi}(2:end);
%                 deltaFs.(currStim){tsi} = siTraces.(currStim){tsi}(:, 2:end);
%             end
%         end
%     end
%         
%     tSI = cat(1, siTimes.(currStim){1:end});
% 
%     interpfunc = @(x) linspace(x(1), x(end), size(tSI,2));
%     mwinterpMat = cell2mat(arrayfun(@(i) interpfunc(tMW(i,:)), 1:size(tMW,1), 'UniformOutput', false)');
%     nFrames = size(mwinterpMat,2);
% 
%     nTrials= size(tMW,1);
%     for trial=1:nTrials
%         nRois = size(deltaFs.(currStim){trial},1);
%         roiInterp = zeros(nRois, nFrames);
%         for roi=1:nRois
%             F = griddedInterpolant(siTimes.(currStim){trial}, deltaFs.(currStim){trial}(roi,:));
%             roiInterp(roi,:) = F(mwinterpMat(2,:));    
%         end
%         iTraces.(currStim){end+1} = roiInterp;    
%     end
% 
%     interpTraceMat = cat(1, iTraces.(currStim){1:end});
%     currRoiTracesInterp = interpTraceMat(currRoi:nRois:end, :);

    % 
%     subplot(1,2,2)
%     dfTraceMat = cat(1, deltaFs.(currStim){1:end});
%     currRoiTraces = dfTraceMat(currRoi:nRois:end, :);
%     for trial=1:size(currRoiTraces, 1);
%         plot(mwinterpMat(trial,:), currRoiTraces(trial,:), 'k', 'LineWidth', 0.2);
%         hold on;
%     end
%     meanTrace = mean(currRoiTraces, 1);
%     %meanMWTime = mean(tMW,1);
%     plot(meanMWTime, meanTrace, 'LineWidth', 2);
%     hold on;


    % ---------------------------------------------------------------------
    % PLOT:
    % ---------------------------------------------------------------------
    subplot(pRows, round(nStim/pRows), pIdx)
    
    % Plot each trial for current stimulus:
    for trial=1:size(dfMat, 1);
        plot(mwinterpMat(trial,:), dfMat(trial,:), 'k', 'LineWidth', 0.2);
        hold on;
    end
    
    % Plot MW stim ON patch:
    stimOnset = mean(tMW(:,2));
    stimOffset = mean(tMW(:,3));
    ylims = get(gca, 'ylim');
    v = [stimOnset ylims(1); stimOffset ylims(1);...
        stimOffset ylims(2); stimOnset ylims(2)];
    f = [1 2 3 4];
    patch('Faces',f,'Vertices',v,'FaceColor','red', 'FaceAlpha', 0.2, 'EdgeColor', 'none')
    hold on;
    
    % Plot MEAN trace for current ROI across stimulus reps:
    meanTraceInterp = mean(dfMat, 1);
    meanMWTime = mean(mwinterpMat,1);
    plot(meanMWTime, meanTraceInterp,'LineWidth', 2);
    hold on;
    title(sprintf('Stim ID: %s', currStim));
    
    %plot(mean(mwinterpMat,1), zeros(1,length(mean(mwinterpMat,1))), '.'), 
    
    pIdx = pIdx + 1;

end


%%

interpolant = griddedInterpolant(gridcell, siTimes.stim1{1}.');


for trial=1:length(siTraces.stim1)
    siTraces.stim1{trial}(20,:);
    
fStim1 = cat(1,siTraces.stim1{1:end});


roi=1;
testmat = zeros(size(tSI));
for i=1:size(siTraces.stim1, 2)
    
    testmat(i,:) = siTraces.stim1{i}(roi,:);
end

%%

for sidx = slicesToUse
    currSlice = trialStruct.slice(sidx).sliceIdx;
    
    roi = 20;
    for stimIdx=1:length(condTypes)
        currStim = condTypes{stimIdx};
        
        currDfs = trialStruct.slice(sidx).deltaFs.(currStim);
        currTimes = trialStruct.slice(sidx).siTimes.(currStim);
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