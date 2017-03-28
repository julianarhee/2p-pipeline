function avgimg = getCurrentSliceImage(handles, D)


selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedFile = handles.runMenu.Value;

% Get avg image for selected slice and file:
traceStruct = load(fullfile(D.guiPrepend, D.tracesPath, D.traceNames{selectedSliceIdx}));
avgimg = traceStruct.file(selectedFile).avgImage;
%avgimg = avgimg{1};


end