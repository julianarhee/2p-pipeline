function avgimg = getCurrentSliceImage(handles, D)


selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedFile = handles.runMenu.Value;

% Get avg image for selected slice and file:
traceStruct = load(fullfile(D.guiPath, 'traces', D.traceNames{selectedSliceIdx}));
avgimg = traceStruct.avgImage.file(selectedFile);
avgimg = avgimg{1};


end