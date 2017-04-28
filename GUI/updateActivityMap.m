function [handles, D] = updateActivityMap(handles, D, meta)

fprintf('Updating activity map...');

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
if selectedSliceIdx > length(D.slices)
    selectedSliceIdx = length(D.slices);
end
selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
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
    dfStruct = load(fullfile(D.guiPrepend, D.outputDir, D.dfStructName));
    setappdata(handles.roigui, 'df', dfStruct);
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
    displayMap = mapStruct.file(selectedFile).(selectedMapType);
    magMap = mapStruct.file(selectedFile).ratio;

    thresholdMap = threshold_map(displayMap, magMap, currThresh);
end

switch selectedMapType
    
    case 'phase'
        thresholdMap = threshold_map(displayMap, magMap, currThresh);
        axes(handles.ax2);  
        if isfield(D, 'tefo') && D.tefo
            handles.map = imagesc(fov);
            hold on;
            handles.map = imagesc2(thresholdMap);
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
        
    case 'phasemax'
        thresholdMap = threshold_map(displayMap, magMap, currThresh);
        axes(handles.ax2);  
        if D.tefo
            handles.map = imagesc(fov);
            hold on;
            handles.map = imagesc2(thresholdMap);
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
            if isfield(handles.retinolegend, 'Children')
                handles.retinolegend.Children.Visible = 'off';
            end
        end

    case 'phaseInferred'
        % recalculate ratio masks from inferred trace data:
        magMap = mapStruct.file(selectedFile).ratioInferred;
        thresholdMap = threshold_map(displayMap, magMap, currThresh);
        axes(handles.ax2);  
        if isfield(D, 'tefo') && D.tefo
            handles.map = imagesc(fov);
            hold on;
            handles.map = imagesc2(thresholdMap);
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

        
    otherwise
        % 'ratio' 
        % 'magnitude'
        % 'maxDf'
        axes(handles.ax2);  
        if isfield(D, 'tefo') && D.tefo
            handles.map = imagesc(fov);
            hold on;
            thresholdMap = threshold_map(displayMap, displayMap, currThresh);
            handles.map = imagesc2(thresholdMap); %, handles.ax2);
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
            caxis([min(thresholdMap(:)), max(thresholdMap(:))]);
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
%colorbar();

fprintf('...Done!\n');


end