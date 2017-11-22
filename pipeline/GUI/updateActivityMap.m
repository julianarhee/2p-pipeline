function [handles, D] = updateActivityMap(handles, D, meta)

orient = false; %true;

fprintf('Updating activity map...');

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
if selectedSliceIdx > length(D.slices)
    selectedSliceIdx = length(D.slices);
end
selectedSlice = D.slices(selectedSliceIdx) % - D.slices(1) + 1;
selectedFile = handles.runMenu.Value;

currThresh = str2double(handles.threshold.String);

%avgimg = getCurrentSliceImage(handles, D);
fov = repmat(mat2gray(D.avgimg), [1, 1, 3]);

currRunName = meta.file(selectedFile).mw.runName;
runpts= strsplit(currRunName, '_');
currCondType = runpts{1};

% mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
% if ~exist(fullfile(D.guiPrepend, D.outputDir, mapStructName), 'file')
%     mapTypes = {'NaN'};
% else     
%     mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
%     mapTypes = fieldnames(mapStruct.file(selectedFile));
% end

% Populate MAP menu options: --------------------------------------------
if isfield(D, 'dfStructName')
    dfStruct = getappdata(handles.roigui, 'df'); %load(fullfile(D.guiPrepend, D.outputDir, D.dfStructName));
    %setappdata(handles.roigui, 'df', dfStruct);
    if isempty(dfStruct.slice(selectedSlice).file)
        fprintf('No DF struct found for slice %i.\n', selectedSlice);
        noDF = true;
    else
        noDF = false;
    end
else
    fprintf('No DF struct found in current acquisition.\n');
    noDF = true;
end

mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
if ~exist(fullfile(D.guiPrepend, D.outputDir, mapStructName), 'file')
    if noDF
        mapTypes = {'NaN'};
    else
        mapTypes = {'maps - notfound'};
    end
else
    mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
    mapTypes = fieldnames(mapStruct.file(selectedFile));
end
if noDF %isempty(dfStruct.slice(selectedSlice).file)
    mapTypes{end+1} = 'maxDf - not found';
else
    mapTypes{end+1} = 'maxDf';
end
handles.mapMenu.String = mapTypes;
    


if handles.mapMenu.Value > length(mapTypes)
    handles.mapMenu.Value = length(mapTypes);
end
selectedMapIdx = handles.mapMenu.Value;

selectedMapType = mapTypes{selectedMapIdx};
if noDF
    displayMap = zeros(size(fov));
elseif any(strfind(selectedMapType, 'Df')) || any(strcmp(selectedMapType, 'maps - notfound'))
    displayMap = dfStruct.slice(selectedSlice).file(selectedFile).maxMap;
    if ~strcmp(selectedMapType, 'maxDf')
        handles.mapMenu.Value = find(ismember(mapTypes, 'maxDf'));
        selectedMapIdx = handles.mapMenu.Value;
        selectedMapType = mapTypes{selectedMapIdx};
    end
else
%     displayMap = mapStruct.file(selectedFile).(selectedMapType);
%     magMap = mapStruct.file(selectedFile).ratio;
%     thresholdMap = threshold_map(displayMap, magMap, currThresh);
    
    tcourseTypes = handles.timecourseMenu.String;
    selected_tcourse = handles.timecourseMenu.Value;
%     switch tcourseTypes{selected_tcourse}
%         case 'processedNMF'
%             if ~isempty(strfind(selectedMapType, 'output'))
%                 selectedMapType = strcat(selectedMapType(1:end-9), 'NMF');
%                 updateMapMenu = true;
%             elseif isempty(strfind(selectedMapType, 'NMF'))
%                 selectedMapType = strcat(selectedMapType, 'NMF');
%                 updateMapMenu = true;
%             else
%                 updateMapMenu = false;
%             end
%         case 'dfNMF'
%             if ~isempty(strfind(selectedMapType, 'NMF')) && isempty(strfind(selectedMapType, 'output'))
%                 selectedMapType = strcat(selectedMapType(1:end-3), 'NMFoutput');
%                 updateMapMenu = true;
%             elseif isempty(strfind(selectedMapType, 'NMF'))
%                 selectedMapType = strcat(selectedMapType, 'NMFoutput');
%                 updateMapMenu = true;
%             else
%                 updateMapMenu = false;
%             end
%         otherwise
%             updateMapMenu = false;
%     end
    if ~isempty(strfind(tcourseTypes{selected_tcourse}, '- NMF'))
        %case 'processedNMF'
            magMap = mapStruct.file(selectedFile).ratioNMF;
            if ~isempty(strfind(selectedMapType, 'output'))
                selectedMapType = strcat(selectedMapType(1:end-9), 'NMF');
                updateMapMenu = true;
            elseif isempty(strfind(selectedMapType, 'NMF'))
                selectedMapType = strcat(selectedMapType, 'NMF');
                updateMapMenu = true;
            else
                updateMapMenu = false;
            end
    elseif ~isempty(strfind(tcourseTypes{selected_tcourse}, 'output'))
            magMap = mapStruct.file(selectedFile).ratioNMFoutput;
            if ~isempty(strfind(selectedMapType, 'NMF')) && isempty(strfind(selectedMapType, 'output'))
                selectedMapType = strcat(selectedMapType(1:end-3), 'NMFoutput');
                updateMapMenu = true;
            elseif isempty(strfind(selectedMapType, 'NMF'))
                selectedMapType = strcat(selectedMapType, 'NMFoutput');
                updateMapMenu = true;
            else
                updateMapMenu = false;
            end
        %otherwise
    elseif isempty(strfind(tcourseTypes{selected_tcourse}, 'NMF'))
        if ~isempty(strfind(selectedMapType, 'output'))
            selectedMapType = selectedMapType(1:end-9);
            updateMapMenu = true;
        elseif ~isempty(strfind(selectedMapType, 'NMF'))
            selectedMapType = selectedMapType(1:end-3);
            updateMapMenu = true;
        else
            updateMapMenu = false;
        end
        magMap = mapStruct.file(selectedFile).ratio;
    else
        magMap = mapStruct.file(selectedFile).ratio;
        updateMapMenu = false;
    
    end
    
    if updateMapMenu
        handles.mapMenu.Value = find(cell2mat(cellfun(@(maptype) strcmp(maptype, selectedMapType), mapTypes, 'UniformOutput', 0)));
        selectedMapIdx = handles.mapMenu.Value;
    end
    displayMap = mapStruct.file(selectedFile).(selectedMapType);
    %magMap = mapStruct.file(selectedFile).ratio;
    thresholdMap = threshold_map(displayMap, magMap, currThresh);
end

%switch selectedMapType
    
if strfind(selectedMapType, 'phase')
    thresholdMap = threshold_map(displayMap, magMap, currThresh);
    axes(handles.ax2);  
    if isfield(D, 'tefo') && D.tefo
        if orient
            handles.map = imagesc(rot90(fliplr(fov), -2));
            hold on;
            handles.map = imagesc2(rot90(fliplr(thresholdMap), -2));
        else
            handles.map = imagesc(fov);
            hold on;
            handles.map = imagesc2(thresholdMap);
        end
    else
        handles.map = imagesc(scalefov(fov));
        hold on;
        handles.map = imagesc2(scalefov(thresholdMap));
    end
    colormap(handles.ax2, hsv);
    %caxis([min(displayMap(:)), max(displayMap(:))]);
    caxis([-1*pi, pi]);
    colorbar off;

    %legend:
    if isfield(meta, 'legends')
        legends = meta.legends;
        axes(handles.retinolegend)
        handles.retinolegend.Visible = 'on';
        handles.maplegend = imagesc(legends.(currCondType));
        axis off
        colormap(handles.retinolegend, hsv);
        %caxis([min(displayMap(:)), max(displayMap(:))]);
        caxis([-1*pi, pi]);
        colorbar off;

    else
        fprintf('No legend found...\n');
        handles.retinolegend.Visible = 'off';
    end
        
else
    % 'ratio' 
    % 'magnitude'
    % 'maxDf'
    axes(handles.ax2);  
    if isfield(D, 'tefo') && D.tefo
        if orient
            handles.map = imagesc(rot90(fliplr(fov), -2));
            hold on;
            thresholdMap = threshold_map(displayMap, displayMap, currThresh);
            handles.map = imagesc2(rot90(fliplr(thresholdMap), -2)); %, handles.ax2);
        else
            handles.map = imagesc(fov);
            hold on;
            thresholdMap = threshold_map(displayMap, displayMap, currThresh);
            handles.map = imagesc2(thresholdMap); %, handles0
        end
    else
        handles.map = imagesc(scalefov(fov));
        hold on;
        thresholdMap = threshold_map(displayMap, displayMap, currThresh);
        handles.map = imagesc2(scalefov(thresholdMap)); %, handles.ax2);    
    end
    %handles.map = imagesc2(scalefov(displayMap)); %, handles.ax2);
    colormap(handles.ax2, hot);
    %caxis([min(displayMap(:)), max(displayMap(:))]);
    if any(isnan([min(thresholdMap(:)), max(thresholdMap(:))]))
        caxis([0 1]);
    else
        if min(thresholdMap(:))==max(thresholdMap(:))
            caxis([0, max(thresholdMap(:))]);
        else
            caxis([min(thresholdMap(:)), max(thresholdMap(:))]);
        end
    end
    colorbar();

    axes(handles.retinolegend)
    handles.retinolegend.Visible = 'off';
    if ~isempty(handles.retinolegend.Children)
        handles.retinolegend.Children.Visible = 'off';
    end
end

refPos = handles.ax1.Position;
ax2Pos = handles.ax2.Position;
handles.ax2.Position(3:4) = [refPos(3:4)];
title(handles.ax2, currRunName);

% If BAR, run=file=stimulus, so update stim-plot:
if strcmp(D.stimType, 'bar')
    condtypes = handles.stimMenu.String;
    condmatches = find(cellfun(@(c) any(strfind(c, currRunName)), condtypes));
    if length(find(condmatches))>1
        % Get corresponding FILE:
        findidx = regexp(currRunName, '[\d]');
        if isempty(findidx)
            stimMenuIdx = condmatches(1);
        else
            stimMenuIdx = condmatches(findidx);
        end
    else
        stimMenuIdx = find(condmatches);
    end
    handles.stimMenu.Value = stimMenuIdx;
end
%colorbar();

fprintf('...Done!\n');


end