function updateStimulusPlot(handles, D)

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedSlice = D.slices(selectedSliceIdx);
selectedFile = handles.runMenu.Value;
selectedStimIdx = handles.stimMenu.Value;
stimNames = handles.stimMenu.String;
selectedStim = handles.stimMenu.String{selectedStimIdx};

selectedRoi = str2double(handles.currRoi.String);


stimstruct = getappdata(handles.roigui, 'stimstruct');
stimcolors = getappdata(handles.roigui, 'stimcolors');

dfMat = stimstruct.slice(selectedSlice).(selectedStim).dfCell{selectedRoi}.*100;
mwSec = stimstruct.slice(selectedSlice).(selectedStim).mwinterpMat;
mwTrialTimes = stimstruct.slice(selectedSlice).(selectedStim).mwTrialTimes;


%PLOT:
axes(handles.ax3);

if ~handles.stimShowAvg.Value % Show each trial:
%     for trial=1:size(dfMat, 1);
%         trial
%         plot(mwinterpMat(trial,:), dfMat(trial,:), 'Color', [0.7 0.7 0.7], 'LineWidth',0.1);
%         hold on;
%     end
    
    % Plot each trial trace:
    handles.stimplot.trials = plot(mwSec.', dfMat.', 'Color', [0.7 0.7 0.7], 'LineWidth',0.1);
    hold on;
    
    % Plot MW stim ON patch:
    stimOnset = mean(mwTrialTimes(:,2));
    stimOffset = mean(mwTrialTimes(:,3));
    ylims = get(gca, 'ylim');
    v = [stimOnset ylims(1); stimOffset ylims(1);...
        stimOffset ylims(2); stimOnset ylims(2)];
    f = [1 2 3 4];
    handles.stimplot.mwepochs = patch('Faces',f,'Vertices',v,'FaceColor',stimcolors(selectedStimIdx,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on;
    
    % Plot MEAN trace for current ROI across stimulus reps:
    meanTraceInterp = mean(dfMat, 1);
    meanMWTime = mean(mwSec,1);
    handles.stimplot.mean = plot(meanMWTime, meanTraceInterp, 'k', 'LineWidth', 2);
    hold on;
    title(sprintf('Stim ID: %s', selectedStim));
    
else
    
    for stimIdx=1:length(stimNames)
        stimname = ['stim' num2str(stimIdx)];
        dfMat = stimstruct.slice(selectedSlice).(stimname).dfCell{selectedRoi}.*100;
        mwSec = stimstruct.slice(selectedSlice).(stimname).mwinterpMat;
        mwTrialTimes = stimstruct.slice(selectedSlice).(stimname).mwTrialTimes;
        
        meanTraceInterp = mean(dfMat, 1);
        meanMWTime = mean(mwSec,1);
    
        handles.stimplot.mean(stimIdx) = plot(meanMWTime, meanTraceInterp, 'Color', stimcolors(stimIdx,:), 'LineWidth', 2);
        hold on;
        
    end
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


handles.ax3.Box = 'off';
handles.TickDir = 'out';
hold off;

%end

% 
% axes(handles.ax4);
% handles.timecourse = plot(; %, handles.ax2);
% colormap(handles.ax2, hot);
% caxis([min(displayMap(:)), max(displayMap(:))]);
% colorbar();
% handles.retinolegend.Visible = 'off';
% 
% 
% refPos = handles.ax1.Position;
% ax2Pos = handles.ax2.Position;
% handles.ax2.Position(3:4) = [refPos(3:4)];
%title(currRunName);
%colorbar();
%
end