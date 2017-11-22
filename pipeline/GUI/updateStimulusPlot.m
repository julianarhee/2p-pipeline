function [handles, D] = updateStimulusPlot(handles, D)

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
if selectedSliceIdx > length(D.slices)
    selectedSliceIdx = length(D.slices);
end
selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
selectedFile = handles.runMenu.Value;
if strcmp(D.stimType, 'bar') && length(handles.runMenu.String)==length(handles.stimMenu.String)
    if handles.stimMenu.Value ~= selectedFile
        handles.stimMenu.Value = selectedFile;
    end
end
selectedStimIdx = handles.stimMenu.Value;
stimNames = handles.stimMenu.String;
selectedStim = handles.stimMenu.String{selectedStimIdx};

selectedRoi = str2double(handles.currRoi.String);

guicolors = getappdata(handles.roigui, 'guicolors');

%PLOT:
axes(handles.ax3);
if strcmp(D.stimType, 'bar')
    % do some other stuff
    
    fftStructName = sprintf('fft_Slice%02d.mat', selectedSlice); %D.slices(selectedSliceIdx)); 
    fftstruct = load(fullfile(D.guiPrepend, D.outputDir, fftStructName));
    freqs = fftstruct.file(selectedFile).freqs;
    tcourseTypes = handles.timecourseMenu.String;
    selected_tcourse = handles.timecourseMenu.Value;
%     switch tcourseTypes{selected_tcourse}
%         case 'processedNMF'
%             mags = fftstruct.file(selectedFile).magMatNMF(:, selectedRoi);
%         case 'dfNMF'
%             mags = fftstruct.file(selectedFile).magMatNMFoutput(:, selectedRoi);
%         otherwise
%             mags = fftstruct.file(selectedFile).magMat(:, selectedRoi);
%     end
    if ~isempty(strfind(tcourseTypes{selected_tcourse}, '- NMF'))
        mags = fftstruct.file(selectedFile).magMatNMF(:, selectedRoi);
    elseif ~isempty(strfind(tcourseTypes{selected_tcourse}, 'NMFoutput'))
        mags = fftstruct.file(selectedFile).magMatNMFoutput(:, selectedRoi);
    else
        mags = fftstruct.file(selectedFile).magMat(:, selectedRoi);
    end
    targetfreqIdx = fftstruct.file(selectedFile).targetFreqIdx;
    targetfreq = fftstruct.file(selectedFile).targetFreq;
    handles.fft = plot(freqs, mags, 'k', 'LineWidth', 2);
    hold on
    handles.fftMax = plot(freqs(mags==max(mags(:))), mags(mags==max(mags(:))), 'r*');
    hold on;
    handles.fftTarget = plot(freqs(targetfreqIdx), mags(targetfreqIdx), 'g*');
    xlabel('Frequency (Hz)')
    ylabel('Magnitude')
    hold off;
else
    
stimstruct = getappdata(handles.roigui, 'stimstruct');
stimcolors = getappdata(handles.roigui, 'stimcolors');

trialstruct = getappdata(handles.roigui, 'trialstruct');

% dfMat = stimstruct.slice(selectedSlice).(selectedStim).dfCell{selectedRoi}.*100;
% mwSec = stimstruct.slice(selectedSlice).(selectedStim).mwinterpMat;
mwTrialTimes = stimstruct.slice(selectedSlice).(selectedStim).mwTrialTimes;
% nTrials = size(dfMat,1);

dfMat = stimstruct.slice(selectedSlice).(selectedStim).dfTraceCell{selectedRoi}.*100;
%dfMat = stimstruct.slice(selectedSlice).(selectedStim).dfCell{selectedRoi}.*100;
mwSec = stimstruct.slice(selectedSlice).(selectedStim).siTimeMat; % If created before 03/29/2017, need to transpose
trimmed = stimstruct.slice(selectedSlice).(selectedStim).trimmedAndPadded;
nTrials = size(dfMat,2);

if ~handles.stimShowAvg.Value % Show each trial:
    
    % Plot each df/f trace for all trials:
    if handles.smooth.Value
        for tridx=1:size(dfMat, 2)
            dfMat(:,tridx) = smooth(dfMat(:,tridx), 'rlowess');
        end
    end
        
    handles.stimtrials = plot(mwSec, dfMat, 'Color', guicolors.lightgray, 'LineWidth',0.1);
    setappdata(handles.roigui, 'dfMat', dfMat);          
    hold on;
    
    % Plot MW stim ON patch:
    stimOnset = mean(mwTrialTimes(:,2));
    stimOffset = mean(mwTrialTimes(:,3));
    ylims = get(gca, 'ylim');
    v = [stimOnset ylims(1); stimOffset ylims(1);...
        stimOffset ylims(2); stimOnset ylims(2)];
    f = [1 2 3 4];
    handles.stimepochs = patch('Faces',f,'Vertices',v,'FaceColor',stimcolors(selectedStimIdx,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on;
    
    % Plot MEAN trace for current ROI across stimulus reps:
    %meanTraceInterp = mean(dfMat, 1);
    meanTraceInterp = nanmean(dfMat, 2);
    meanMWTime = nanmean(mwSec,2);
    handles.stimtrialmean = plot(meanMWTime, meanTraceInterp, 'k', 'LineWidth', 2);
    hold on;
else
    meanDfMat = {};
    for stimIdx=1:length(stimNames)
        stimname = ['stim' num2str(stimIdx)];
        %dfMat = stimstruct.slice(selectedSlice).(stimname).dfCell{selectedRoi}.*100;
        %mwSec = stimstruct.slice(selectedSlice).(stimname).mwinterpMat;
        mwTrialTimes = stimstruct.slice(selectedSlice).(stimname).mwTrialTimes;

        dfMat = stimstruct.slice(selectedSlice).(stimname).dfTraceCell{selectedRoi}.*100;
        if handles.smooth.Value
            for tridx=1:size(dfMat, 2)
                dfMat(:,tridx) = smooth(dfMat(:,tridx), 'rlowess');
            end
        end
    
        %dfMat = stimstruct.slice(selectedSlice).(stimname).dfCell{selectedRoi}.*100;
        mwSec = stimstruct.slice(selectedSlice).(stimname).siTimeMat; %
        trimmed = stimstruct.slice(selectedSlice).(stimname).trimmedAndPadded;
        
        %meanTraceInterp = mean(dfMat, 1);
        meanTraceInterp = nanmean(dfMat, 2);
        meanMWTime = nanmean(mwSec,2);
    
        handles.stimtrialmean(stimIdx) = plot(meanMWTime, meanTraceInterp, 'Color', stimcolors(stimIdx,:), 'LineWidth', 0.5);
        hold on;
        
        meanDfMat{end+1} = meanTraceInterp;
        
    end
    setappdata(handles.roigui, 'dfMat', meanDfMat);     
           
            
    % Plot MW stim ON patch:
    stimOnset = mean(mwTrialTimes(:,2));
    stimOffset = mean(mwTrialTimes(:,3));
    ylims = get(gca, 'ylim');
    v = [stimOnset ylims(1); stimOffset ylims(1);...
        stimOffset ylims(2); stimOnset ylims(2)];
    f = [1 2 3 4];
    handles.stimplot.mwepochs = patch('Faces',f,'Vertices',v,'FaceColor','black', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on;
    
end

end

handles.ax3.Box = 'off';
handles.ax3.TickDir = 'out';
hold off;


end


