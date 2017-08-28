function [handles, D] = updateTimeCourse(handles, D, meta)

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
if selectedSliceIdx > length(D.slices)
    selectedSliceIdx = length(D.slices);
end
selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
selectedFile = handles.runMenu.Value;

selectedRoi = str2double(handles.currRoi.String);

%fov = repmat(mat2gray(D.avgimg), [1, 1, 3]);

currRunName = meta.file(selectedFile).mw.runName;
runpts= strsplit(currRunName, '_');
currCondType = runpts{1};
%volumeIdxs = selectedSlice:meta.file(selectedFile).si.nFramesPerVolume:meta.file(selectedFile).si.nTotalFrames;
volumeIdxs = D.slices(selectedSliceIdx):meta.file(selectedFile).si.nFramesPerVolume:meta.file(selectedFile).si.nTotalFrames;
% 


dfstruct = getappdata(handles.roigui, 'df');
if ~isempty(fieldnames(dfstruct))
    if isempty(dfstruct.slice(selectedSlice).file)
        fprintf('No DF struct found for slice %i.\n', selectedSlice);
        noDF = true;
    else
        noDF = false;
    end
else
    fprintf('No DF struct found in current acquisition.\n');
    noDF = true;
end

axes(handles.ax4);
    
tstamps = meta.file(selectedFile).mw.siSec(volumeIdxs);
stimStarts = meta.file(selectedFile).mw.stimStarts;
mwTimes = meta.file(selectedFile).mw.mwSec;

tcourseTypes = handles.timecourseMenu.String;
if handles.timecourseMenu.Value > length(tcourseTypes)
    handles.timecourseMenu.Value = length(tcourseTypes);
end
switch tcourseTypes{handles.timecourseMenu.Value}
    case 'dF/F'
        if ~noDF
            dfMat = dfstruct.slice(selectedSlice).file(selectedFile).dfMat;
%             if isfield(handles, 'timecourse')
%                 handles.timecourse.CData = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
%             else 
            if isempty(dfMat)
                handles.timecourse = plot(tstamps(1:size(dfMat,1)), zeros(size(dfMat,1)), 'k', 'LineWidth', 1);
                hold on;
            else
                handles.timecourse = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
            end
            xlim([0 tstamps(size(dfMat,1))]); % Crop extra white space
            hold on;
        else
            handles.timecourse = plot(tstamps(1:size(dfMat,1)), zeros(1,1:length(tstamps)), 'k', 'LineWidth', 1);
            hold on;
        end
        xlabel('time (s)');
        ylabel('dF/f');
        
%     case 'inferred'
%         if ~noDF
%             dfMat = dfstruct.slice(selectedSlice).file(selectedFile).dfMatInferred;
% %             if isfield(handles, 'timecourse')
% %                 handles.timecourse.CData = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
% %             else 
%             handles.timecourse = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
% %             end
%             xlim([0 tstamps(size(dfMat,1))]); % Crop extra white space
%             hold on;
%         else
%             handles.timecourse = plot(tstamps, zeros(1,1:length(tstamps)), 'k', 'LineWidth', 1);
%             hold on;
%         end
%         xlabel('time (s)');
%         ylabel('dF/f');
        
    case 'df/F - NMF'
        if ~noDF
            dfMat = dfstruct.slice(selectedSlice).file(selectedFile).dfMatNMF;
%             if isfield(handles, 'timecourse')
%                 handles.timecourse.CData = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
%             else 
            if isempty(dfMat)
                handles.timecourse = plot(tstamps(1:size(dfMat,1)), zeros(size(dfMat,1)), 'k', 'LineWidth', 1);
            else
                handles.timecourse = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
            end

            xlim([0 tstamps(size(dfMat,1))]); % Crop extra white space
            hold on;
        else
            handles.timecourse = plot(tstamps, zeros(1,1:length(tstamps)), 'k', 'LineWidth', 1);
            hold on;
        end
        xlabel('time (s)');
        ylabel('dF/f');
        
    case 'dfNMFoutput'
        if ~noDF
            dfMat = dfstruct.slice(selectedSlice).file(selectedFile).dfMatNMFoutput;
%             if isfield(handles, 'timecourse')
%                 handles.timecourse.CData = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
%             else 
            if isempty(dfMat)
                handles.timecourse = plot(tstamps(1:size(dfMat,1)), zeros(size(dfMat,1)), 'k', 'LineWidth', 1);
            else
                handles.timecourse = plot(tstamps(1:size(dfMat,1)), dfMat(:,selectedRoi), 'k', 'LineWidth', 1);
            end
            xlim([0 tstamps(size(dfMat,1))]); % Crop extra white space
            hold on;
        else
            handles.timecourse = plot(tstamps, zeros(size(tstamps)), 'k', 'LineWidth', 1);
            hold on;
        end
        xlabel('time (s)');
        ylabel('dF/f');
        
        
    case 'raw'
        % laod traces raw
        tracestruct = load(fullfile(D.tracesPath, D.traceNames{selectedSliceIdx}));
        rawmat = tracestruct.file(selectedFile).rawTraces;
%         if isfield(handles, 'timecourse')
%             handles.timecourse.CData = plot(tstamps, rawmat(:,selectedRoi), 'k', 'LineWidth', 1);
%         else
        if isempty(rawmat)
            handles.timecourse = plot(tstamps(1:size(rawmat,1)), zeros(size(rawmat,1)), 'k', 'LineWidth', 1);
        else
            handles.timecourse = plot(tstamps, rawmat(:,selectedRoi), 'k', 'LineWidth', 1);
        end
%         end
        xlim([0 tstamps(end)]);
        hold on;
        xlabel('time (s)');
        ylabel('intensity');

    case 'processed'
        % laodtracemat
        tracestruct = load(fullfile(D.tracesPath, D.traceNames{selectedSliceIdx}));
        if ~isfield(tracestruct.file(selectedFile), 'traceMatDC')
            tracemat = tracestruct.file(selectedFile).traceMat;
        else
            tracemat = tracestruct.file(selectedFile).traceMatDC;
        end
%         if isfield(handles, 'timecourse')
%             handles.timecourse.CData = plot(tstamps(1:size(tracemat,1)), tracemat(:,selectedRoi), 'k', 'LineWidth', 1);
%         else
        if isempty(tracemat)
            handles.timecourse = plot(tstamps(1:size(tracemat,1)), zeros(size(tracemat,1)), 'k', 'LineWidth', 1);
        else
            handles.timecourse = plot(tstamps(1:size(tracemat,1)), tracemat(:,selectedRoi), 'k', 'LineWidth', 1);
        end
        xlim([0 tstamps(end)]); % Crop extra white space
        hold on;
        xlabel('time (s)');
        ylabel('intensity');
        
    case 'raw - NMF'
        % laod traces raw
        tracestruct = load(fullfile(D.tracesPath, D.traceNames{selectedSliceIdx}));
        rawmat = tracestruct.file(selectedFile).rawTracesNMF;
%         if isfield(handles, 'timecourse')
%             handles.timecourse.CData = plot(tstamps, rawmat(:,selectedRoi), 'k', 'LineWidth', 1);
%         else
        if isempty(rawmat)
            handles.timecourse = plot(tstamps(1:size(rawmat,1)), zeros(size(rawmat,1)), 'k', 'LineWidth', 1);
        else
            handles.timecourse = plot(tstamps, rawmat(:,selectedRoi), 'k', 'LineWidth', 1);
        end
        xlim([0 tstamps(end)]);
        hold on;
        xlabel('time (s)');
        ylabel('intensity');

    case 'processed - NMF'
        % laodtracemat
        tracestruct = load(fullfile(D.tracesPath, D.traceNames{selectedSliceIdx}));
        tracemat = tracestruct.file(selectedFile).rawTraceMatDCNMF;
%         if isfield(handles, 'timecourse')
%             handles.timecourse.CData = plot(tstamps(1:size(tracemat,1)), tracemat(:,selectedRoi), 'k', 'LineWidth', 1);
%         else
        if isempty(tracemat)
            handles.timecourse = plot(tstamps(1:size(tracemat,1)), zeros(size(tracemat,1)), 'k', 'LineWidth', 1);
        else
            handles.timecourse = plot(tstamps(1:size(tracemat,1)), tracemat(:,selectedRoi), 'k', 'LineWidth', 1);
        end
        xlim([0 tstamps(end)]); % Crop extra white space
        hold on;
        xlabel('time (s)');
        ylabel('intensity');
        
        
end

ylims = get(gca,'ylim');
if strcmp(D.stimType, 'bar')
    for cyc=1:length(stimStarts)
        x = meta.file(selectedFile).mw.mwSec(stimStarts(cyc));
        handles.mwepochs(cyc) = line([x x], [ylims(1) ylims(2)], 'Color', [1 0 0 0.5]);
        handles.ax4.TickDir = 'out';
        hold on;
    end
    nEpochs = length(stimStarts);
else
    colors = getappdata(handles.roigui, 'stimcolors');
    
    mwCodes = meta.file(selectedFile).mw.pymat.(currRunName).stimIDs;
    sy = [ylims(1) ylims(1) ylims(2) ylims(2)];
    trialidx = 1;
    currStimTrialIdx = [];
    for trial=1:2:length(mwTimes)
        sx = [mwTimes(trial) mwTimes(trial+1) mwTimes(trial+1) mwTimes(trial)];
        currStim = mwCodes(trial);
        if handles.stimShowAvg.Value
            handles.mwepochs(trial) = patch(sx, sy, colors(currStim,:,:), 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
        else
            if currStim==handles.stimMenu.Value
                handles.mwepochs(trial) = patch(sx, sy, colors(currStim,:,:), 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
                currStimTrialIdx = [currStimTrialIdx trialidx];
            else
                handles.mwepochs(trial) = patch(sx, sy, [0.7 0.7 0.7], 'FaceAlpha', 0.3, 'EdgeAlpha', 0);
            end
        end
        handles.ax4.TickDir = 'out';
        hold on;
        trialidx = trialidx + 1;
        %handles.ax4.UserData.trialEpochs = trialidx;
    end
    nEpochs = length(mwTimes);
    
%     % Darken selected trial if multiple reps of selected stim in current
%     % run:
%     if isfield(handles.ax3.UserData, 'clickedTrialIdxInRun')
%         handles.mwepochs(currStimTrialIdx(handles.ax3.UserData.clickedTrialIdxInRun)).FaceAlpha = selectedTrialStimAlpha;
%     end
    
end

handles.ax4.UserData.trialEpochs = nEpochs;
% xlim([0 tstamps(size(dfMat,2))]); % Crop extra white space
% xlabel('time (s)');

handles.ax4.Box = 'off';
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
title(currRunName);
%colorbar();
%
end