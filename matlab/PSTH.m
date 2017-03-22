% PSTHs.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

acquisition_info;

slicesToUse = D.slices;

%%
% -------------------------------------------------------------------------
% Process traces for TRIAL analysis:
% -------------------------------------------------------------------------
tic()
for sidx = 1:length(slicesToUse)

    currSlice = slicesToUse(sidx);
    fprintf('Processing traces for Slice %02d...\n', currSlice);
    
    % Assumes all TIFFs are reps of e/o, so just use file1:
    winUnit = 3; % single trial dur (sec) 
    %crop = meta.file(1).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
    processTraces(D, winUnit)

end
tic()
dfMin = 20;
fprintf('Getting df structs for each movie file...\n');
getDfMovie(D, dfMin);
toc()

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
nTiffs = M.nTiffs;

% Get trace struct names:
% traceNames = dir(fullfile(D.tracesPath, '*.mat'));
% traceNames = {traceNames(:).name}';

slicesToUse = D.slices;

skipFirst = true;
ITI = 2;

trialstruct = struct();
for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    
    T = load(fullfile(D.tracesPath, D.traceNames{sidx}));

    mwTrials = struct();
    mwEpochIdxs = struct();
    siEpochIdxs = struct();
    siTrials = struct();
    siTraces = struct();
    siTimes = struct();
    mwTimes = struct();
    deltaFs = struct();
    stmiInfo = struct();
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
        stimInfo.(currStim) = [];
    end
    
    % Get MW tstamps and within-trial indices ("epochs") and sort by
    % stimulus-type and run number (i.e., tiff file):
    % ---------------------------------------------------------------------
    for fidx=1:nTiffs
        
        meta = M.file(fidx);
        volumeIdxs = currSlice:meta.si.nFramesPerVolume:meta.si.nTotalFrames;

        currRun = meta.mw.runName;
        currMWcodes = meta.mw.pymat.(currRun).stimIDs;
        currTrialIdxs = meta.mw.pymat.(currRun).idxs + 1; % bc python 0-indexes
        
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
    end 
    
    % Find closest matching frame tstamps for each trial across runs, and
    % sort by stimulus-type:
    % ---------------------------------------------------------------------
    tic()
    nTrials = struct();
    for stimIdx=1:length(condTypes) 

        currStim = condTypes{stimIdx};
        nTrials.(currStim) = 0;

        if ~isfield(mwTrials, currStim)
            fprintf('File %s not found for stim %s\n', currTiff, currStim);
            continue;
        end
            
        currTiffs = fieldnames(mwTrials.(currStim));
        for tiffIdx=1:length(currTiffs)
            
%             % If corrected traceMat not found, correct and save, otherwise
%             % just load corrected tracemat:
%             if ~isfield(T, 'traceMat') || length(T.traceMat.file) < tiffIdx
%                 
%                 % 1. First, remove "bad" frames (too much motion), and
%                 % replace with Nans:
%                 currTraces = T.rawTraces.file{tiffIdx};
%                 T.badFrames.file{tiffIdx}(T.badFrames.file{tiffIdx}==T.refframe.file(tiffIdx)) = []; % Ignore reference frame (corrcoef=1)
%                 
%                 bf = T.badFrames.file{tiffIdx};
%                 if length(bf) > 1
%                     assignNan = @(f) nan(size(f));
%                     tmpframes = arrayfun(@(i) assignNan(currTraces(:,i)), bf, 'UniformOutput', false);
%                     currTraces(:,bf) = cat(2,tmpframes{1:end});
%                 end
%                 
%                 % 2. Next, get subtract rolling average from each trace
%                 % (each row of currTraces is the trace of an ROI).
%                 % Interpolate NaN frames if there are any.
%                 winsz = round(meta.si.siVolumeRate*3*2);
%                 [traceMat, DCs] = arrayfun(@(i) subtractRollingMean(currTraces(i, :), winsz), 1:size(currTraces, 1), 'UniformOutput', false);
%                 traceMat = cat(1, traceMat{1:end});
%                 DCs = cat(1, DCs{1:end});
%                 traceMat = bsxfun(@plus, DCs, traceMat);
%                 T.traceMat.file{tiffIdx} = traceMat;
%                 T.winsz.file(tiffIdx) = winsz;
%                 T.DCs.file{fidx} = DCs;
%                 save(fullfile(D.tracesPath, D.traceNames{sidx}), '-append', '-struct', 'T');
%             else
%                 traceMat = T.traceMat.file{tiffIdx};
%             end
            traceMat = T.traceMat.file{tiffIdx};
            
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
                    trialinfo.stimName = currStim;
                    trialinfo.runName = currTiffs{tiffIdx};
                    trialinfo.trialIdxInRun = trialIdx;
                    trialinfo.currStimRep = nTrials.(currStim);
                    stimInfo.(currStim){end+1} = trialinfo;
                    
                    % Calculate dFs:
                    %ctraces = traceMat(:, frameidxs);
                    %baselines = mean(traceMat(:,frameidxs(1):frameOn-1), 2);
                    %baselineMat = repmat(baselines, [1, size(ctraces,2)]);
                    %deltaFs.(currStim){end+1} = ((ctraces - baselineMat) ./ baselineMat)*100;
                end
            end
        end
            
    end
    toc()
    
    trialstruct.slice(currSlice).sliceIdx = currSlice;
    trialstruct.slice(currSlice).mwTrials = mwTrials;
    trialstruct.slice(currSlice).siTrials = siTrials;
    trialstruct.slice(currSlice).siTraces = siTraces;
    trialstruct.slice(currSlice).siTimes = siTimes;
    trialstruct.slice(currSlice).mwTimes = mwTimes;
    trialstruct.slice(currSlice).mwEpochIdxs = mwEpochIdxs;
    trialstruct.slice(currSlice).siEpochIdxs = siEpochIdxs;
    trialstruct.slice(currSlice).deltaFs = deltaFs;
    trialstruct.slice(currSlice).nTrials = nTrials;
    trialstruct.slice(currSlice).info = stimInfo;
end

trialstructName = 'stimReps.mat';
save_struct(D.outputDir, trialstructName, trialstruct);

D.trialStructName = trialstructName;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');


%%

stimstruct = struct();

for sliceIdx=1:length(D.slices)
    currSlice = D.slices(sliceIdx);

    %iTraces = struct();
    %rTraces = struct();
    
    %nRois = size(siTraces.stim1{1}, 1);
    mwTimes = trialstruct.slice(currSlice).mwTimes;
    siTimes = trialstruct.slice(currSlice).siTimes;
    siTraces = trialstruct.slice(currSlice).siTraces;

    nStim = length(fieldnames(siTraces));
    %pIdx = 1;
    for stimIdx=1:length(fieldnames(siTraces))
        currStim = sprintf('stim%i', stimIdx);
        iTraces.(currStim) = [];
        rTraces.(currStim) = [];

        tMW = cat(1, mwTimes.(currStim){1:end});
        nTrials= size(tMW,1);
        
        currnpoints = [];
        for tsi=1:length(siTimes.(currStim))
            currnpoints = [currnpoints length(siTimes.(currStim){tsi})];
        end
        nFrames = min(currnpoints); %nFrames = size(mwinterpMat,2);
        
        %tSI = cat(1,siTimes.(currStim){1:end});

        interpfunc = @(x) linspace(x(1), x(end), nFrames);
        mwinterpMat = cell2mat(arrayfun(@(i) interpfunc(tMW(i,:)), 1:size(tMW,1), 'UniformOutput', false)');

    
        % Create cell array containing a cell for each ROI:
        % Each cell in this array will have that ROI's traces for each 
        % trial of the current stimulus ("currStim"):
        nRois = size(siTraces.(currStim){1}, 1);
        roivec = 1:nRois; 
        rawTraceCell = arrayfun(@(xval) cellfun(@(c) c(xval,:), siTraces.(currStim), 'UniformOutput', false), roivec, 'UniformOutput',false);
        % - length(rawTraceMat) = nRois
        % - length(rawTraceMat{roiNo}) = # total trials for currStim
        % - i.e., each cell in this array contains the traces for each
        % trial of a given ROI.
        
        % Get interpolated traces for 
        
        %tic()
        trialvec = 1:nTrials;
        interpTraceCell = arrayfun(@(roi) arrayfun(@(trial) interpolateTraces(siTimes.(currStim){trial}, rawTraceCell{roi}{trial}, mwinterpMat(trial,:)), trialvec, 'UniformOutput', false), roivec, 'UniformOutput', false);
        %toc()

        %
        
        stimstruct.slice(currSlice).(currStim).mwinterpMat = mwinterpMat;
        stimstruct.slice(currSlice).(currStim).interpTraceCell = interpTraceCell;
        stimstruct.slice(currSlice).(currStim).rawTraceCell = rawTraceCell;
        stimstruct.slice(currSlice).(currStim).nRois = nRois;
        stimstruct.slice(currSlice).(currStim).nFramesPerTrial = currnpoints;
        stimstruct.slice(currSlice).(currStim).nTrials = nTrials;
        stimstruct.slice(currSlice).(currStim).mwTrialTimes = tMW;
    end

end

stimstructName = 'processedTraces.mat';
save_struct(D.outputDir, stimstructName, stimstruct);

D.stimStructName = stimstructName;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');


%%

currSlice = 15;
currStim = 'stim1';

currRoi = 20;
nFramesPerTrial = stimstruct.slice(currSlice).(currStim).nFramesPerTrial;
minFrames = min(nFramesPerTrial);
rawMat = arrayfun(@(i) stimstruct.slice(currSlice).(currStim).rawTraceCell{currRoi}{i}(1:minFrames), 1:length(stimstruct.slice(currSlice).(currStim).rawTraceCell{currRoi}), 'UniformOutput', false);
raw = cat(1, rawMat{1:end});
interp = cat(1, stimstruct.slice(currSlice).(currStim).interpTraceCell{currRoi}{1:end});


%         rawTraceMat = cat(1, rawTraceMat{1}{1:end});
%         interpTraceMat = cat(1, interpTraceCell{1}{1:end});
        
%
figure();
subplot(1,2,1)
plot(raw.', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
plot(mean(raw,1), 'k', 'LineWidth', 1);
title('no interp')

subplot(1,2,2)
plot(interp.', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
plot(mean(interp,1), 'k', 'LineWidth', 1);
title('interp')

%         
figure();
for i=1:nTrials
    subplot(2,9, i)
    plot(raw(i,:), 'k');
    hold on;
    plot(interp(i,:), 'r');
end
suptitle('Interp vs Raw traces by trial');






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


%%
% ---------------------------------------------------------------------
% LOAD:
% ---------------------------------------------------------------------

acquisition_info;
stimstruct = load(fullfile(D.outputDir, D.stimStructName));

%trialstruct = load(fullfile(D.outputDir, D.trialStructName));

%% PLOT:

currSlice = 15;
currRoi = 20;

figure();
pRows = 2;

nStim = length(fieldnames(stimstruct.slice(currSlice)));

for pIdx=1:nStim
    
    currStim = sprintf('stim%i', pIdx);

    %raw = cat(1, stimstruct.slice(currSlice).(currStim).rawTraceCell{currRoi}{1:end});
    interp = cat(1, stimstruct.slice(currSlice).(currStim).interpTraceCell{currRoi}{1:end});
    mwinterpMat = stimstruct.slice(currSlice).(currStim).mwinterpMat;
    tMW = stimstruct.slice(currSlice).(currStim).mwTrialTimes;


    [baseR, baseC] = find(mwinterpMat>=0);
    baselineMatTmp = interp.*(mwinterpMat<0);
    baselineMatTmp(baseR, baseC) = NaN;
    baselines = nanmean(baselineMatTmp,2);
    baselineMat = repmat(baselines, [1, size(interp,2)]);
    dfMat = ((interp - baselineMat) ./ baselineMat)*100;

    subplot(pRows, round(nStim/pRows), pIdx)

    % Plot each trial for current stimulus:
    for trial=1:size(dfMat, 1);
        plot(mwinterpMat(trial,:), dfMat(trial,:), 'Color', [0.7 0.7 0.7], 'LineWidth',0.1);
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
    plot(meanMWTime, meanTraceInterp, 'k', 'LineWidth', 2);
    hold on;
    title(sprintf('Stim ID: %s', currStim));

    %plot(mean(mwinterpMat,1), zeros(1,length(mean(mwinterpMat,1))), '.'), 

    pIdx = pIdx + 1;
%
end

%% LOOK AT DF/f MAPS


DF = load(fullfile(D.outputDir, D.dfStructName));

currSlice = 15;
nMovies = length(DF.slice(currSlice).file);

figure();
pRows = 2;
for tiffidx=1:nMovies
    
    dfstruct = DF.slice(currSlice).file(tiffidx);
    
    h = subplot(pRows, ceil(nMovies/pRows), tiffidx);
    imagesc2(dfstruct.maxMap, h);
    axis off
    colormap(hot)
    colorbar()
    title(sprintf('Movie %i', tiffidx))
    hold on;
    
end

figTitle = strrep(D.acquisitionName, '_', '-');
suptitle(sprintf('Max dF/Fs: %s', figTitle))

%% Look at DF/F traces:

currSlice = 15;
currRoi = 20;

DF = load(fullfile(D.outputDir, D.dfStructName));
M = load(D.metaPath);

nStimuli = length(M.condTypes);
colors = zeros(nStimuli,3);
for c=1:nStimuli
    colors(c,:,:) = rand(1,3);
end
    
nMovies = length(DF.slice(currSlice).file);

figure();
pRows = nMovies;

for tiffidx=1:nMovies
    
    meta = M.file(tiffidx);
    dfstruct = DF.slice(currSlice).file(tiffidx);
    
    volumeIdxs = currSlice:meta.si.nFramesPerVolume:meta.si.nTotalFrames;
    currFrameTimes = meta.mw.siSec(volumeIdxs);
    currMWTimes = meta.mw.mwSec;
    currMWCodes = meta.mw.pymat.(meta.mw.runName).stimIDs;

    currtrace = dfstruct.dfMat(currRoi,:);
    
    subplot(pRows, ceil(nMovies/pRows), tiffidx);
    plot(currFrameTimes, currtrace, 'k')
    ylims = get(gca, 'ylim');
    sy = [ylims(1) ylims(1) ylims(2) ylims(2)];
    hold on;
    for trial=1:2:length(currMWTimes)
        sx = [currMWTimes(trial) currMWTimes(trial+1) currMWTimes(trial+1) currMWTimes(trial)];
        currStim = currMWCodes(trial);
        patch(sx, sy, colors(currStim,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0);
        hold on;
    end
    xlim([0 currFrameTimes(end)]);

end



%%

% %%
% 
% interpolant = griddedInterpolant(gridcell, siTimes.stim1{1}.');
% 
% 
% for trial=1:length(siTraces.stim1)
%     siTraces.stim1{trial}(20,:);
%     
% fStim1 = cat(1,siTraces.stim1{1:end});
% 
% 
% roi=1;
% testmat = zeros(size(tSI));
% for i=1:size(siTraces.stim1, 2)
%     
%     testmat(i,:) = siTraces.stim1{i}(roi,:);
% end
% 
% %%
% 
% for sidx = slicesToUse
%     currSlice = trialstruct.slice(sidx).sliceIdx;
%     
%     roi = 20;
%     for stimIdx=1:length(condTypes)
%         currStim = condTypes{stimIdx};
%         
%         currDfs = trialstruct.slice(sidx).deltaFs.(currStim);
%         currTimes = trialstruct.slice(sidx).siTimes.(currStim);
%         for dfIdx=1:length(currDfs)
%             t = currTimes{dfIdx}; % - currTimes{dfIdx}(1);
%             df = currDfs{dfIdx}(roi,:);
%             plot(t, df, 'k', 'LineWidth', 0.5);
%             hold on;
%         end
%         trialsBySlice = cat(3, currDfs{:});
%         meanTracesCurrSlice = mean(trialsBySlice, 3);
%         plot(t, meanTracesCurrSlice(roi,:), 'k', 'LineWidth', 2);
%         
%         % ---
%         for roi=1:size(meanTracesCurrSlice,1)
%             plot(t, meanTracesCurrSlice(roi,:), 'k', 'LineWidth', .2);
%             hold on;
%         end
%         % ---
% 
%         
%         
% %         for roi=1:size(meanTracesCurrSlice,1)
% %             plot(meanTracesCurrSlice(roi,:));
% %         end
%         
%     end
%     
%     y = T.traces.file{tiffIdx}(roi,:);
%     dff = (y - mean(y))./mean(y);
%     plot(meta.mw.siSec(volumeIdxs), dff);
%     hold on;
%     stimOnsets = meta.mw.mwSec(meta.mw.stimStarts);
%     stimOffsets = meta.mw.mwSec(2:2:end);
%     for onset=1:length(stimOnsets)
%         %line([stimOnsets(onset) stimOnsets(onset)], get(gca, 'ylim'));
%         ylims = get(gca, 'ylim');
%         v = [stimOnsets(onset) ylims(1); stimOffsets(onset) ylims(1);...
%             stimOffsets(onset) ylims(2); stimOnsets(onset) ylims(2)];
%         f = [1 2 3 4];
%         patch('Faces',f,'Vertices',v,'FaceColor','red', 'FaceAlpha', 0.2)
%         hold on;
%     end
%     
%     stimNames = fieldnames(mwTrials);
%     for stimIdx=1:length(stimNames)
%         fprintf('%i trials found for %s.\n', length(fieldnames(mwTrials.(stimNames{stimIdx}))), stimNames{stimIdx});
%     end
%     
%     
%     
%     v = [0 0; 1 0; 1 1; 0 1];
% f = [1 2 3 4];
% patch('Faces',f,'Vertices',v,'FaceColor','red')
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     siTrials = struct();
%     for stimIdx=1:length(condTypes)
%         if ~isfield(mwTrials, condTypes{stimIdx})
%             fprintf('
%         
% 
% 
%             
%             
%             
%     
%     
%     
% end