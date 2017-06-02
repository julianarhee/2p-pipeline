% PSTHs.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

% acquisition_info;
% session = '20161222_JR030W';
% experiment = 'gratings1';
% analysis_no = 4;
session = '20161219_JR030W';
% experiment = 'gratingsFinalMask2';
experiment = 'retinotopyFinalMask';
analysis_no = 8;
tefo = true;

D = loadAnalysisInfo(session, experiment, analysis_no, tefo);

slicesToUse = D.slices;

%
% -------------------------------------------------------------------------
% Process traces for TRIAL analysis:
% -------------------------------------------------------------------------
tic()

winUnit = 3; 
nWinUnits = 3;

switch D.roiType
    case '3Dcnmf'
        processTraces3Dnmf(D, winUnit, nWinUnits)
    case 'cnmf'
        % do other stuff
    otherwise
        processTraces(D, winUnit, nWinUnits)
end
fprintf('Done processing Traces!\n');

% end
% -------------------------------------------------------------------------
% Get DF/F for whole movie:
% -------------------------------------------------------------------------
dfMin = 20;
fprintf('Getting df structs for each movie file...\n');

dfMin = 20;
switch D.roiType
    case '3Dcnmf'
        dfstruct = getDfMovie3Dnmf(D, dfMin);
    case 'cnmf'
        % do other stuff
    otherwise
        dfstruct = getDfMovie(D, dfMin);
end

D.dfStructName = dfstruct.name;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

fprintf('Done extracting DF/F!\n');


% Get colormap:
% legends = makeLegends(D.outputDir);
% meta.legends = legends;
% save(D.metaPath, '-append', '-struct', 'meta');
meta = load(D.metaPath);
nStimuli = length(meta.condTypes);
colors = zeros(nStimuli,3);
for c=1:nStimuli
    colors(c,:,:) = rand(1,3);
end
meta.stimcolors = colors;
save(D.metaPath, '-append', '-struct', 'meta');

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
meta = load(D.metaPath);
condTypes = meta.condTypes;
nTiffs = meta.nTiffs;

% Get trace struct names:
% traceNames = dir(fullfile(D.tracesPath, '*.mat'));
% traceNames = {traceNames(:).name}';

slicesToUse = D.slices;

skipFirst = true;
ITI = 2;

trialstruct = struct();
for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    
    tracestruct = load(fullfile(D.tracesPath, D.traceNames{sidx}));

    mwTrials = struct();
    mwEpochIdxs = struct();
    siEpochIdxs = struct();
    siTrials = struct();
    siTraces = struct();
    siTimes = struct();
    mwTimes = struct();
    deltaFs = struct();
    stimInfo = struct();
    trialIdxs = struct();
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
        trialIdxs.(currStim) = [];
    end
    
    % Get MW tstamps and within-trial indices ("epochs") and sort by
    % stimulus-type and run number (i.e., tiff file):
    % ---------------------------------------------------------------------
    for fidx=1:nTiffs
        
        %meta = meta.file(fidx);
        volumeIdxs = currSlice:meta.file(fidx).si.nFramesPerVolume:meta.file(fidx).si.nTotalFrames;

        currRun = meta.file(fidx).mw.runName;
        currMWcodes = meta.file(fidx).mw.pymat.(currRun).stimIDs;
        currTrialIdxs = meta.file(fidx).mw.pymat.(currRun).idxs + 1; % bc python 0-indexes
        
        currTiff = sprintf('tiff%i', fidx);
        trialNum = 1;
        for trialIdx=1:2:length(currMWcodes)
            if skipFirst && trialIdx==1
                continue;
            end
            if trialIdx < currTrialIdxs(end)
                currEnd = trialIdx + 2;
            else
                currEnd = length(meta.file(fidx).mw.mwSec);
            end
            currStim = sprintf('stim%i', currMWcodes(trialIdx));
            if ~isfield(mwTrials, currStim)
                mwTrials.(currStim).(currTiff){1} = meta.file(fidx).mw.mwSec(trialIdx-1:currEnd);
            else
                if ~isfield(mwTrials.(currStim), currTiff)
                    mwTrials.(currStim).(currTiff){1} = meta.file(fidx).mw.mwSec(trialIdx-1:currEnd);
                else
                    mwTrials.(currStim).(currTiff){end+1} = meta.file(fidx).mw.mwSec(trialIdx-1:currEnd);
                end
            end
            mwEpochIdxs.(currStim){end+1} = trialIdx-1:currEnd;
            if ~isfield(trialIdxs, currStim)
                trialIdxs.(currStim).(currTiff){1} = trialNum;
            else
                if ~isfield(trialIdxs.(currStim), currTiff)
                    trialIdxs.(currStim).(currTiff){1} = trialNum;
                else
                    trialIdxs.(currStim).(currTiff){end+1} = trialNum;
            
                end
            end
            trialNum = trialNum + 1;
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
              currTiff = currTiffs{tiffIdx};
              currTiffIdx = str2double(currTiff(5:end));
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
            traceMat = tracestruct.file(currTiffIdx).traceMatDC;
            
            currTrials = mwTrials.(currStim).(currTiff);
            for trialIdx=1:length(currTrials)
                currSecs = mwTrials.(currStim).(currTiff){trialIdx};
                tPre = currSecs(1) + 1;
                tOn = currSecs(2);
%                 if length(currSecs) < 3
%                     continue;
%                     tOff = tOn + 1;
%                 else
%                     tOff = currSecs(3);
%                 end
                tOff = currSecs(3);
                if length(currSecs) < 4
                    continue;
                    fprintf('Not enough tstamps found in stim %s, file %i, trial %i.\n', currStim, tiffIdx, trialIdx);
                    tPost = tOff + ITI;
                else
                    tPost = currSecs(4);
                end
                %end
                    %sliceSecs = meta.file(currTiffIdx).mw.siSec(volumeIdxs);
                    if length(meta.file(currTiffIdx).si.siFrameTimes) < length(volumeIdxs)
                        sliceSecs =  meta.file(currTiffIdx).mw.siSec(volumeIdxs);
                    else
                        sliceSecs = meta.file(currTiffIdx).si.siFrameTimes(volumeIdxs);
                    end
                    
                    
                    framePre = find(abs(sliceSecs-tPre) == min(abs(sliceSecs-tPre)));
                    frameOn = find(abs(sliceSecs-tOn) == min(abs(sliceSecs-tOn)));
                    frameOff = find(abs(sliceSecs-tOff) == min(abs(sliceSecs-tOff)));
                    framePost = find(abs(sliceSecs-tPost) == min(abs(sliceSecs-tPost)));
                    frameEpochIdxs = [framePre frameOn frameOff framePost];
                    siTrials.(currStim).(currTiff){trialIdx} = sliceSecs(frameEpochIdxs);

                    nTrials.(currStim) = nTrials.(currStim) + 1;
                    frameidxs = frameEpochIdxs(1):frameEpochIdxs(end);
    %                 if length(frameidxs)>23
    %                     frameidxs = frameidxs(1:23);
    %                 end
                    %siTraces.(currStim){end+1} = traceMat(:, frameidxs);
                    siTraces.(currStim){end+1} = traceMat(frameidxs,:);
                    siTimes.(currStim){end+1} = sliceSecs(frameidxs) - tOn;
                    %roiTimes.(currStim){end+1} = sliceSecs(frameidxs);
                    mwTimes.(currStim){end+1} = [tPre tOn tOff tPost] - tOn; %currSecs - tOn;
                    siEpochIdxs.(currStim){end+1} = frameEpochIdxs;
                    %siOnsetTime.(currStim){end+1} = sliceSecs(tOn)
                    trialinfo.stimName = currStim;
                    trialinfo.runName = currTiff; %currTiffs{tiffIdx};
                    trialinfo.tiffNum = currTiffIdx;
                    trialinfo.trialIdxInRun = trialIdxs.(currStim).(currTiff){trialIdx};
                    trialinfo.currStimRep = nTrials.(currStim);
                    stimInfo.(currStim){end+1} = trialinfo;
                    
                    % Calculate dFs:
                    %ctraces = traceMat(:, frameidxs);
                    %baselines = mean(traceMat(:,frameidxs(1):frameOn-1), 2);
                    %baselineMat = repmat(baselines, [1, size(ctraces,2)]);
                    %deltaFs.(currStim){end+1} = ((ctraces - baselineMat) ./ baselineMat)*100;
                %end
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

%
% %% Don't interpolate -- 
% % Create same-sized time and trace mats using MW-onset-matched SI-times
% % (i.e., frame times from SI), and fill in or remove frame tstamps on a
% % trial-by-trial basis:
% 
% %stimtracestruct = struct();
% %siTraces.stim3{1}(:,1)==stimstruct.slice(currSlice).stim3.rawTraceCell{1}{1}
% 
% for sliceIdx=1:length(D.slices)
%     
%     currSlice = D.slices(sliceIdx);
%     siTimes = trialstruct.slice(currSlice).siTimes;
%     siTraces = trialstruct.slice(currSlice).siTraces;
%     
%     stimnames = fieldnames(stimstruct.slice(currSlice));
%     nStim = length(stimnames);
%     for stimIdx=1:nStim
%         
%         currStim = stimnames{stimIdx};
%         
%         % Most trials will have the same nFrames, but some will have 1 or 2
%         % more/less, depending on where the closest-matching indices were.
%         % 
%         nTrials = length(siTimes.(currStim));
%         nRois = size(siTraces.(currStim){1},2);
%         nFramesPerTrial = cellfun(@length, siTimes.(currStim));
%         nTrialsDifferent = length(unique(nFramesPerTrial));
% 
%         if nTrialsDifferent > 1
%             nFramesStandard = mode(nFramesPerTrial);
%             if nFramesStandard < max(nFramesPerTrial)-2
%                 nFramesStandard = median(nFramesPerTrial);
%             end
% 
%             % First, trim trials with frame tstamps extending into the next
%             % trial onset:
%             trialsTooLong = arrayfun(@(i) siTimes.(currStim){i}(end) > winUnit, 1:nTrials);
%             trimmedTrialTimes = arrayfun(@(i) siTimes.(currStim){i}(1:end-1), find(trialsTooLong), 'UniformOutput', false);
%             trimmedTrialTraces = arrayfun(@(i) siTraces.(currStim){i}(1:end-1,:), find(trialsTooLong), 'UniformOutput', false);
% 
%             siTimes.(currStim)(trialsTooLong) = trimmedTrialTimes(1:end);
%             siTraces.(currStim)(trialsTooLong) = trimmedTrialTraces(1:end);
% 
%     %         % Second, pad trials with shifted onset times relative to the
%     %         % standard in current set:
%     %         findOnsetIdxs = cellfun(@(trial) find(trial>0,1), siTimes.(currStim));
%     %         onsetIdxStandard = mode(findOnsetIdxs);
%     %         trialsTooEarly = find(findOnsetIdxs<onsetIdxStandard)
%     %         paddedTimes = arrayfun(@(t) padarray(siTimes.(currStim){t}, [0 onsetIdxStandard-findOnsetIdxs(t)], 0, 'pre'), trialsTooEarly, 'UniformOutput', false)
%     %         paddedTraces = arrayfun(@(t) padarray(siTraces.(currStim){t}, [onsetIdxStandard-findOnsetIdxs(t) 0], 0, 'pre'), trialsTooEarly, 'UniformOutput', false)
%     %         
%     %         siTimes.(currStim)(trialsTooEarly) = paddedTimes(1:end);
%     %         siTraces.(currStim)(trialsTooEarly) = paddedTraces(1:end);
%     %         
%             % Second, pad trials that are "too short" with to create same-sized
%             % mat:
%             trialsTooShort = arrayfun(@(trial) length(siTimes.(currStim){trial})<nFramesStandard, 1:nTrials);
%             trialsTooShortIdxs = find(trialsTooShort);
%             paddedTimes = arrayfun(@(t) padarray(siTimes.(currStim){t}, [0 nFramesStandard-length(siTimes.(currStim){t})], 0, 'post'), trialsTooShortIdxs, 'UniformOutput', false);
%             paddedTraces = arrayfun(@(t) padarray(siTraces.(currStim){t}, [nFramesStandard-length(siTimes.(currStim){t}) 0], 0, 'post'), trialsTooShortIdxs, 'UniformOutput', false);
% 
%             siTimes.(currStim)(trialsTooShortIdxs) = paddedTimes(1:end);
%             siTraces.(currStim)(trialsTooShortIdxs) = paddedTraces(1:end);
%             
%             trimmedAndPadded = true;
%         else
%             trimmedAndPadded = false;
% 
%         end
%         siTimeMat = cat(1, siTimes.(currStim){1:end});
%         siTimeMat(siTimeMat==0) = NaN; % Each ROW is 1 time-course for a trial
%         
%         siTraceCellTmp = arrayfun(@(roi) cellfun(@(trial) trial(:,roi), siTraces.(currStim), 'UniformOutput', false), 1:nRois, 'UniformOutput',false);
%         siTraceCell = cellfun(@(roi) cat(2,roi{1:end}), siTraceCellTmp, 'UniformOutput', false);
% 
%         siTraceCell = cellfun(@nanzero, siTraceCell, 'UniformOutput', false);
%         
%         % ** Each CELL contains trials for an roi: MxN mat, each column in
%         % N is 1 trial's traces, so each row corresponds to a point in time
%         % (frame time point).
%         % To plot all traces, plot(siTraceCell{ROI}), since plots
%         % column-wise.
%         % To plot MEAN of traces, plot(mean(siTraceCell{ROI},2)), since
%         % columns=trials and rows=tpoints, want to take mean across trials
%         % for each tpoint.
%         
%         nFrames = size(siTimeMat,2);
%         nTrials = size(siTimeMat,1);
%         
%         preidxs = siTimeMat < 0;
%         tmpBaselines = cellfun(@(roi) preidxs.'.*roi, siTraceCell, 'UniformOutput', false);
%         baselines = cellfun(@(roi) arrayfun(@(trial) mean(roi(roi(:,trial)>0, trial)), 1:nTrials), tmpBaselines, 'UniformOutput', false);
%         baselines = cat(1,baselines{1:end}); % nr = nROIS, nc = ntrials
%         baselines = baselines.'; % baseline value for EACH trial is in row --
%         % --> for ROI x, pull baselines{:,x} where : grabs all the trials
%         
%         nRois = size(baselines,2);
%         calcdffunc = @(x,y) (x-y)./y;
%         dfTraceCell = arrayfun(@(roi) calcdffunc(siTraceCell{roi}, repmat(baselines(:,roi),1,size(siTraceCell{roi},1)).'), 1:nRois, 'UniformOutput', false);
% 
%         
%         stimstruct.slice(currSlice).(currStim).siTimeMat = siTimeMat.'; 
%         % --> transpose siTimeMat to get each COLUMN to correspond to
%         % tstamps of a single trial...
%         stimstruct.slice(currSlice).(currStim).siTraceCell = siTraceCell;
%         stimstruct.slice(currSlice).(currStim).dfTraceCell = dfTraceCell;
%         stimstruct.slice(currSlice).(currStim).trimmedAndPadded = trimmedAndPadded;
% %         
% %         if max(nFramesPerTrial) > nFramesStandard
% %             trialsWithMax = find(nFramesPerTrial==max(nFramesPerTrial));
% %             trialsWithMaxTooLong = trialsWithMax(arrayfun(@(i) siTimes.(currStim){i}(end) > winUnit, trialsWithMax));
% %             % Remove trials that are "too long" (i.e., last frame is
% %             % actually occuring at onset of next trial):
% %             trimmedTrials = arrayfun(@(i) siTimes.(currStim){i}(1:end-1), trialsWithMaxTooLong, 'UniformOutput', false);
% %             siTimes.(currStim)(trialsWithMaxTooLong) = trimmedTrials(1:end);
% % 
% %             % Remove trials if they are too "early" (i.e., onset frame
% %             % is shifted (later in time) relative to the rest of the
% % 
% %         end
%     end
% end
% 
% 
% save(fullfile(D.outputDir, D.stimStructName), '-append', '-struct', 'stimstruct');
% 
% D.trialStructName = trialstructName;
% save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
% 



% Interpolate same-sized time and trace mats with MW times...

stimstruct = struct();

fprintf('--------------------------------------------------\n')
fprintf('1.  Parsing traces into TRIALS...\n');
fprintf('--------------------------------------------------\n')

for sliceIdx=1:length(D.slices)
    currSlice = D.slices(sliceIdx);

    %iTraces = struct();
    %rTraces = struct();
    
    %nRois = size(siTraces.stim1{1}, 1);
    mwTimes = trialstruct.slice(currSlice).mwTimes;
    siTimes = trialstruct.slice(currSlice).siTimes;
    siTraces = trialstruct.slice(currSlice).siTraces;
    mwEpochIdxs = trialstruct.slice(currSlice).mwEpochIdxs;
    

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
        % --> mwinterpMat is nxm mat, n=trial, m=frame-tpoints
    
        % Create cell array containing a cell for each ROI:
        % Each cell in this array will have that ROI's traces for each 
        % trial of the current stimulus ("currStim"):

        tic()
%         rawTraceCell = arrayfun(@(roi) cellfun(@(trial) trial(:,roi), siTraces.(currStim), 'UniformOutput', false),...
%                         1:nRois, 'UniformOutput',false);
        nRoisByTrial = cellfun(@(trial) size(trial,2), siTraces.(currStim));
        if length(unique(nRoisByTrial))>1
            nRoisMax = max(nRoisByTrial);
            paddedRawTraceCell = cellfun(@(trial) padarray(trial, [0 nRoisMax-size(trial,2)], 0, 'post'), siTraces.(currStim), 'UniformOutput', false);
            paddedRawTraceCell = cellfun(@nanzero, paddedRawTraceCell, 'UniformOutput', false);
            rawTraceCell = arrayfun(@(roi) cellfun(@(trial) trial(:,roi), paddedRawTraceCell, 'UniformOutput', false), 1:nRoisMax, 'UniformOutput', false);
            % --> needed to do this for auto-ROIs (cnmf) bec each trial can
            % have different # of ROIs -- just use max nRois, s.t. trials in
            % which there are less than max nRois should be a trace of NaNs.
        else
            nRois = size(siTraces.(currStim){1},2);
            roivec = 1:nRois; 
            rawTraceCell = arrayfun(@(roi) cellfun(@(c) c(:,roi), siTraces.(currStim), 'UniformOutput', false), roivec, 'UniformOutput',false);
        end
    
        toc()
        % - length(rawTraceMat) = nRois
        % - length(rawTraceMat{roiNo}) = # total trials for currStim
        % - i.e., each cell in this array contains the traces for each
        % trial of a given ROI.
        
        % Get interpolated traces for 
        
        tic()
        trialvec = 1:nTrials;
        interpTraceCell = arrayfun(@(roi) arrayfun(@(trial) interpolateTraces(siTimes.(currStim){trial}, rawTraceCell{roi}{trial}, mwinterpMat(trial,:)), trialvec, 'UniformOutput', false), roivec, 'UniformOutput', false);
        toc()

        %
        
        stimstruct.slice(currSlice).(currStim).mwinterpMat = mwinterpMat;
        stimstruct.slice(currSlice).(currStim).interpTraceCell = interpTraceCell;
        stimstruct.slice(currSlice).(currStim).rawTraceCell = rawTraceCell;
        stimstruct.slice(currSlice).(currStim).nRois = nRois;
        stimstruct.slice(currSlice).(currStim).nFramesPerTrial = currnpoints;
        stimstruct.slice(currSlice).(currStim).nTrials = nTrials;
        stimstruct.slice(currSlice).(currStim).mwTrialTimes = tMW;
        
        fprintf('Parsed traces for stim %s, slice %i.\n', currStim, currSlice);
        
    end

end

% stimstructName = 'processedTraces.mat';
% save_struct(D.outputDir, stimstructName, stimstruct);
% 
% D.stimStructName = stimstructName;
% save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

% ------------------------------
fprintf('--------------------------------------------------\n')
fprintf('2.  Getting DF/F for interpolated traces...\n');
fprintf('--------------------------------------------------\n')
% Subtract baseline for df/F:

for sliceIdx=1:length(D.slices)
    
    currSlice = D.slices(sliceIdx);
    
    stimnames = fieldnames(stimstruct.slice(currSlice));
    
    for stimIdx=1:length(stimnames)
        currStim = stimnames{stimIdx};
        mwinterpMat = stimstruct.slice(currSlice).(currStim).mwinterpMat;
        interpTraceCell = stimstruct.slice(currSlice).(currStim).interpTraceCell;
        % Each cell contains trials for given ROI...

        interpTraces = cellfun(@(roi) cat(1,roi{1:end}), interpTraceCell, 'UniformOutput', false);
        % Each cell contains trials in a mat for each ROI:  mxn mat,
        % m=ntrials, n=nframes. Should be the same size as mwinterpMat...
        
        preidxs = mwinterpMat < 0;
        tmpBaselines = cellfun(@(roi) preidxs.*roi, interpTraces, 'UniformOutput', false);
        baselines = cellfun(@(roi) arrayfun(@(trial) mean(roi(trial, roi(trial,:)>0)), 1:size(roi,1)), tmpBaselines, 'UniformOutput', false);
        baselines = cat(1,baselines{1:end}); % nr = nROIS, nc = nframes
        baselines = baselines.'; 
        
        nRois = size(baselines,2);
        calcdffunc = @(x,y) (x-y)./y;
        df = arrayfun(@(roi) bsxfun(calcdffunc, interpTraces{roi}, baselines(:,roi)), 1:nRois, 'UniformOutput', false);
        %df = cellfun(@(roi) roi.', df, 'UniformOutput', false);
        % --> each cell contains mxn mat : m=frames, n=trials
        % --> each cell in df array belongs to a given ROI
        stimstruct.slice(currSlice).(currStim).dfCell = df;              % df{1} contains MxN mat:  M (each row) is a trial, and N (cols) is a frame/tpoint.
        stimstruct.slice(currSlice).(currStim).baselineCell = baselines; % B{1} contains T values, each value is mean of baseline period for trial T
        
        fprintf('Finished getting DF/F for %s, slice %i.\n', currStim, currSlice);
        
    end
end


% save(fullfile(D.outputDir, D.stimStructName), '-append', '-struct', 'stimstruct');
% 
% D.stimStructName = stimstructName;
% save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
% 
% fprintf('Done!\n');


% -----------------------------------------
% Don't interpolate -- 

fprintf('--------------------------------------------------\n')
fprintf('3.  Getting DF/F for NON-interpolated traces...\n');
fprintf('--------------------------------------------------\n')

% Create same-sized time and trace mats using MW-onset-matched SI-times
% (i.e., frame times from SI), and fill in or remove frame tstamps on a
% trial-by-trial basis:

%stimtracestruct = struct();
%siTraces.stim3{1}(:,1)==stimstruct.slice(currSlice).stim3.rawTraceCell{1}{1}
tic()

for sliceIdx=1:length(D.slices)
    
    currSlice = D.slices(sliceIdx);

    siTimes = trialstruct.slice(currSlice).siTimes;
    siTraces = trialstruct.slice(currSlice).siTraces;
    
    stimnames = fieldnames(stimstruct.slice(currSlice));
    nStim = length(stimnames);
    for stimIdx=1:nStim
        
        currStim = stimnames{stimIdx};
        
        roiRawTraceCell = stimstruct.slice(currSlice).(currStim).rawTraceCell;
        
        % Use 1 ROI's trial structure since should be the same for all ROIs
        % of a given stimulus:
        nTrials = length(siTimes.(currStim));
        nFramesPerTrial = cellfun(@length, siTimes.(currStim));
        nTrialsDifferent = length(unique(nFramesPerTrial));

        % Most trials will have the same nFrames, but some will have 1 or 2
        % more/less, depending on where the closest-matching indices were.
        % 
%         nTrials = length(siTimes.(currStim));
%         nRois = size(siTraces.(currStim){1},2);
%         nFramesPerTrial = cellfun(@length, siTimes.(currStim));
%         nTrialsDifferent = length(unique(nFramesPerTrial));

        if nTrialsDifferent > 1
            nFramesStandard = mode(nFramesPerTrial);
            if nFramesStandard < max(nFramesPerTrial)-2
                nFramesStandard = median(nFramesPerTrial);
            end

            % 1.  First, trim trials with frame tstamps extending into the next
            % trial onset:
            trialsTooLong = arrayfun(@(i) siTimes.(currStim){i}(end) > winUnit, 1:nTrials);
            trimmedTrialTimes = arrayfun(@(i) siTimes.(currStim){i}(1:end-1), find(trialsTooLong), 'UniformOutput', false);
            %trimmedTrialTraces = arrayfun(@(i) siTraces.(currStim){i}(1:end-1,:), find(trialsTooLong), 'UniformOutput', false);
            trimmedTrialTraces = cellfun(@(roiTrials) arrayfun(@(t) roiTrials{t}(1:end-1,:), find(trialsTooLong), 'UniformOutput', false), roiRawTraceCell, 'UniformOutput', false);

            siTimes.(currStim)(trialsTooLong) = trimmedTrialTimes(1:end);
            %siTraces.(currStim)(trialsTooLong) = trimmedTrialTraces(1:end);
            %roiRawTraceCell{:}(trialsTooLong) = trimmedTrialTraces{1:end}(1:end);
            roiTrimmedTraceCell = arrayfun(@(roi) trimRoiTrials(roiRawTraceCell{roi}, trimmedTrialTraces{roi}, trialsTooLong), 1:length(roiRawTraceCell), 'UniformOutput', false);
            

            % 2.  Check to see if removing >3.0 trial-offsets makes dim same:
            nFramesPerTrial2 = cellfun(@(s) length(s), siTimes.(currStim));
            trialsStillTooLong = find(nFramesPerTrial2 > nFramesStandard);
            if any(trialsStillTooLong)
                % TODO:  need better tstamps... but for now, just see if first
                % tpoint in "too-long" remaining trial is > 1.0 seconds before
                % stim onset.  remove, if so:
                if abs(siTimes.(currStim){trialsStillTooLong}(1)) > 1
                    trimmedTrialTimes2 = arrayfun(@(i) siTimes.(currStim){i}(2:end), trialsStillTooLong, 'UniformOutput', false);
                    trimmedTrialTraces2 = cellfun(@(roiTrials) arrayfun(@(t) roiTrials{t}(1:end-1,:), trialsStillTooLong, 'UniformOutput', false), roiTrimmedTraceCell, 'UniformOutput', false);
                end
                siTimes.(currStim)(trialsStillTooLong) = trimmedTrialTimes2(1:end);
                roiTrimmedTraceCell = arrayfun(@(roi) trimRoiTrials(roiTrimmedTraceCell{roi}, trimmedTrialTraces2{roi}, trialsStillTooLong), 1:length(roiTrimmedTraceCell), 'UniformOutput', false);
            end
                
                
            % Second, pad trials that are "too short" with to create same-sized
            % mat:
            trialsTooShort = arrayfun(@(trial) length(siTimes.(currStim){trial})<nFramesStandard, 1:nTrials);
            if ~any(trialsTooShort)
                roiPaddedTraceCell = roiTrimmedTraceCell;  % no need to pad
                %continue;
            else
                %trialsTooShortIdxs = find(trialsTooShort);
                paddedTimes = arrayfun(@(t) padarray(siTimes.(currStim){t}, [0 nFramesStandard-length(siTimes.(currStim){t})], 0, 'post'), find(trialsTooShort), 'UniformOutput', false);
                %paddedTraces = arrayfun(@(t) padarray(siTraces.(currStim){t}, [nFramesStandard-length(siTimes.(currStim){t}) 0], 0, 'post'), trialsTooShortIdxs, 'UniformOutput', false);
                paddedTraces = cellfun(@(roiTrials) arrayfun(@(t) padarray(roiTrials{t}, [nFramesStandard-length(roiTrials{t}) 0], 0, 'post'), find(trialsTooShort), 'UniformOutput', false), roiTrimmedTraceCell, 'UniformOutput', false);

                siTimes.(currStim)(trialsTooShort) = paddedTimes(1:end);
                %siTraces.(currStim)(trialsTooShortIdxs) = paddedTraces(1:end);
                roiPaddedTraceCell = arrayfun(@(roi) padRoiTrials(roiTrimmedTraceCell{roi}, paddedTraces{roi}, trialsTooShort), 1:length(roiTrimmedTraceCell), 'UniformOutput', false);
            end
            
            % Cat into MAT:
            roiRawTraceMats = cellfun(@(roi) cat(2, roi{1:end}), roiPaddedTraceCell, 'UniformOutput', false);
            % --> Now, each column is a TRIAL, each row is a frame for curr
            % roi's trace on that trial).
            % --> From Step 1 above, each CELL contains 

            trimmedAndPadded = true;
        else
            roiRawTraceMats = cellfun(@(roi) cat(2, roi{1:end}), roiRawTraceCell, 'UniformOutput', false);
            % --> Now, each column is a TRIAL, each row is a frame for curr
            % roi's trace on that trial).
            % --> From Step 1 above, each CELL contains 

            trimmedAndPadded = false;

        end
        
%         roiRawTraceMats = cellfun(@(roi) cat(2, roi{1:end}), roiPaddedTraceCell, 'UniformOutput', false);
%         % --> Now, each column is a TRIAL, each row is a frame for curr
%         % roi's trace on that trial).
%         % --> From Step 1 above, each CELL contains 
%         
        siTimeMat = cat(1, siTimes.(currStim){1:end});
        siTimeMat(siTimeMat==0) = NaN; % Each ROW is 1 time-course for a trial
        
        %siTraceCellTmp = arrayfun(@(roi) cellfun(@(trial) trial(:,roi), siTraces.(currStim), 'UniformOutput', false), 1:nRois, 'UniformOutput',false);
        %siTraceCell = cellfun(@(roi) cat(2,roi{1:end}), siTraceCellTmp, 'UniformOutput', false);
        %siTraceCell = cellfun(@nanzero, siTraceCell, 'UniformOutput', false);
        siTraceCell = cellfun(@nanzero, roiRawTraceMats, 'UniformOutput', false);
        
        % ** Each CELL contains trials for an roi: MxN mat, each column in
        % N is 1 trial's traces, so each row corresponds to a point in time
        % (frame time point).
        % To plot all traces, plot(siTraceCell{ROI}), since plots
        % column-wise.
        % To plot MEAN of traces, plot(mean(siTraceCell{ROI},2)), since
        % columns=trials and rows=tpoints, want to take mean across trials
        % for each tpoint.
        
        nFrames = size(siTimeMat,2);
        nTrials = size(siTimeMat,1);
        
        preidxs = siTimeMat < 0;
        tmpBaselines = cellfun(@(roi) preidxs.'.*roi, siTraceCell, 'UniformOutput', false);
        baselines = cellfun(@(roi) arrayfun(@(trial) mean(roi(roi(:,trial)>0, trial)), 1:nTrials), tmpBaselines, 'UniformOutput', false);
        baselines = cat(1,baselines{1:end}); % nr = nROIS, nc = ntrials
        baselines = baselines.'; % baseline value for EACH trial is in row --
        % --> for ROI x, pull baselines{:,x} where : grabs all the trials
        
        nRois = size(baselines,2);
        calcdffunc = @(x,y) (x-y)./y;
        dfTraceCell = arrayfun(@(roi) calcdffunc(siTraceCell{roi}, repmat(baselines(:,roi),1,size(siTraceCell{roi},1)).'), 1:nRois, 'UniformOutput', false);

        
        stimstruct.slice(currSlice).(currStim).siTimeMat = siTimeMat.'; 
        % --> transpose siTimeMat to get each COLUMN to correspond to
        % tstamps of a single trial...
        stimstruct.slice(currSlice).(currStim).siTraceCell = siTraceCell;
        stimstruct.slice(currSlice).(currStim).dfTraceCell = dfTraceCell;
        stimstruct.slice(currSlice).(currStim).trimmedAndPadded = trimmedAndPadded;
%         
%         if max(nFramesPerTrial) > nFramesStandard
%             trialsWithMax = find(nFramesPerTrial==max(nFramesPerTrial));
%             trialsWithMaxTooLong = trialsWithMax(arrayfun(@(i) siTimes.(currStim){i}(end) > winUnit, trialsWithMax));
%             % Remove trials that are "too long" (i.e., last frame is
%             % actually occuring at onset of next trial):
%             trimmedTrials = arrayfun(@(i) siTimes.(currStim){i}(1:end-1), trialsWithMaxTooLong, 'UniformOutput', false);
%             siTimes.(currStim)(trialsWithMaxTooLong) = trimmedTrials(1:end);
% 
%             % Remove trials if they are too "early" (i.e., onset frame
%             % is shifted (later in time) relative to the rest of the
% 
%         end
        fprintf('Done processing STIM %s for slice %i.\n', currStim, currSlice);
        
    end
    fprintf('Finished Slice %s!\n', currSlice);
    
end
toc()

%save(fullfile(D.outputDir, D.stimStructName), '-append', '-struct', 'stimstruct');


% Save STIMSTRUCT:
fprintf('Saving STIM STRUCT....\n')

stimstructName = 'processedTraces.mat';
save_struct(D.outputDir, stimstructName, stimstruct);


fprintf('Updating D stuct...\n')

D.stimStructName = stimstructName;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');


D.trialStructName = trialstructName;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

fprintf('DONE!!!\n');

% 
%     mwinterpMat: [MxN double] - M = trial, N = interpolated tpoints to
%     match length of SI traces
% interpTraceCell: {1xnROIs cell} - interpTraceCell{r} = interpolated traces
% for each trial of mwinterpMat for roi r.
%    rawTraceCell: {1xnROIs cell} - rawTraceCell{r} = "raw" (not interpolate)
%    traces for each trial, not matches in tpoints to mwinterpMat
%           nRois: 194
% nFramesPerTrial: [17 17 18 18 18] - should match nframes of
% rawTraceCell{r}
%         nTrials: 5
%    mwTrialTimes: [MxN double] - M = trial, N = mw-based tpoints (trials)
%          dfCell: {1xnROIs cell} - dfCell{r} = df/f calculated from
%          interpolated traces (interpTraceCell), which are matched in
%          tpoints to mwinterpMat (baseline=pre-onset tpoints)
%    baselineCell: [MxnROIs double] - M = trial, gives baseline (average of
%    pre-onset tpoints) for each trial.




%%
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


%     [baseR, baseC] = find(mwinterpMat>=0);
%     baselineMatTmp = interp.*(mwinterpMat<0);
%     baselineMatTmp(baseR, baseC) = NaN;
%     baselines = nanmean(baselineMatTmp,2);
%     baselineMat = repmat(baselines, [1, size(interp,2)]);
%     dfMat = ((interp - baselineMat) ./ baselineMat)*100;

    dfMat = dfstruct.slice(currSlice).(currStim).dfCell{currRoi};
    
    subplot(pRows, round(nStim/pRows), pIdx)

    % Plot each trial for current stimulus:
    figure();
    for trial=1:size(dfMat, 1);
        trial
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


%%
%%

currSlice = 15;
currStim = 'stim1';

currRoi = 20;
nFramesPerTrial = stimstruct.slice(currSlice).(currStim).nFramesPerTrial;
minFrames = min(nFramesPerTrial);
rawMat = arrayfun(@(i) stimstruct.slice(currSlice).(currStim).rawTraceCell{currRoi}{i}(1:minFrames), 1:length(stimstruct.slice(currSlice).(currStim).rawTraceCell{currRoi}), 'UniformOutput', false);
raw = cat(2, rawMat{1:end}); %mxn mat: frame, n=trial
interp = cat(1, stimstruct.slice(currSlice).(currStim).interpTraceCell{currRoi}{1:end});
interp = interp.';

%         rawTraceMat = cat(1, rawTraceMat{1}{1:end});
%         interpTraceMat = cat(1, interpTraceCell{1}{1:end});
        
%
figure();
subplot(1,2,1)
plot(raw, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
plot(mean(raw,2), 'k', 'LineWidth', 1);
title('no interp')

subplot(1,2,2)
plot(interp, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
plot(mean(interp,2), 'k', 'LineWidth', 1);
title('interp')

%         
figure();
for i=1:nTrials
    subplot(2,9, i)
    plot(raw(:,i), 'k');
    hold on;
    plot(interp(:,i), 'r');
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
        trial
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
meta = load(D.metaPath);

nStimuli = length(meta.condTypes);
colors = zeros(nStimuli,3);
for c=1:nStimuli
    colors(c,:,:) = rand(1,3);
end
    
nMovies = length(DF.slice(currSlice).file);

figure();
pRows = nMovies;

for tiffidx=1:nMovies
    
    meta = meta.file(tiffidx);
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
