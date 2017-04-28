function avgimg = getCurrentSliceImage(handles, D)


selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedFile = handles.runMenu.Value;

if selectedSliceIdx > length(D.traceNames)
    selectedSliceIdx = length(D.traceNames);
end

% Get avg image for selected slice and file:
traceStruct = load(fullfile(D.guiPrepend, D.tracesPath, D.traceNames{selectedSliceIdx}));

if iscell(traceStruct.file(selectedFile).avgImage)
    avgimg = traceStruct.file(selectedFile).avgImage{1};
else
    avgimg = traceStruct.file(selectedFile).avgImage;
end

%avgimg = avgimg{1};
while isempty(avgimg)
    for fidx=1:length(traceStruct.file)
        avgimg = traceStruct.file(fidx).avgImage;
        handles.runMenu.Value = fidx;
    end
end

end