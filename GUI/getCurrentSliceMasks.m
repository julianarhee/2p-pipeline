function maskcell = getCurrentSliceMasks(handles,D)

selectedSlice = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedFile = handles.runMenu.Value;

[mp,mn,me] = fileparts(D.maskInfo.maskPaths{selectedSlice});
sliceMaskName = strcat(mn,me);
maskStruct = load(fullfile(D.guiPrepend, mp, sliceMaskName));
fprintf('Loading masks...\n');

if isfield(maskStruct, 'file')
    maskcell = maskStruct.file(selectedFile).maskcell; % masks created with create_rois.m
else
    maskcell = maskStruct.maskcell;
end



end