function updateActivityMap(handles, D, meta)

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedSlice = D.slices(selectedSliceIdx);
selectedFile = handles.runMenu.Value;


%avgimg = getCurrentSliceImage(handles, D);
fov = repmat(mat2gray(D.avgimg), [1, 1, 3]);

currRunName = meta.file(selectedFile).mw.runName;

mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
if ~exist(fullfile(D.guiPrepend, D.outputDir, mapStructName), 'file')
    mapTypes = {'NaN'};
else     
    mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
    mapTypes = fieldnames(mapStruct.file(selectedFile));
end

selectedMapIdx = handles.mapMenu.Value;
selectedMapType = mapTypes{selectedMapIdx};
if ismember(mapTypes, 'NaN')
    displayMap = zeros(size(fov));
else
    displayMap = mapStruct.file(selectedFile).(selectedMapType);
    magMap = mapStruct.file(selectedFile).magnitude;

    currThresh = str2double(handles.threshold.String);
    thresholdMap = threshold_map(displayMap, magMap, currThresh);
end

switch selectedMapType
    
    case 'phase'
        thresholdMap = threshold_map(displayMap, magMap, currThresh);
        axes(handles.ax2);  
        handles.map = imagesc(handles.ax2, fov);
        hold on;
        handles.map = imagesc2(thresholdMap, handles.ax2);
        colormap(handles.ax2, hsv);
        caxis([min(displayMap(:)), max(displayMap(:))]);
        colorbar off;
        
    case 'phasemax'
        thresholdMap = threshold_map(displayMap, magMap, currThresh);
        axes(handles.ax2);  
        handles.map = imagesc(handles.ax2, fov);
        hold on;
        handles.map = imagesc2(thresholdMap, handles.ax2);
        colormap(handles.ax2, hsv);
        caxis([min(displayMap(:)), max(displayMap(:))]);
        colorbar off;

        
    otherwise
        % 'ratio' 
        % 'magnitude'
        % 'maxDf'
        axes(handles.ax2);  
        handles.map = imagesc2(displayMap, handles.ax2);
        colormap(handles.ax2, hot);
        caxis([min(displayMap(:)), max(displayMap(:))])
        colorbar();

    
end
refPos = handles.ax1.Position;
ax2Pos = handles.ax2.Position;
handles.ax2.Position(3:4) = [refPos(3:4)];
title(currRunName);
%colorbar();
%
end