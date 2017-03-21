function masks = getCurrentSliceMasks(handles,D)

selectedSlice = handles.currSlice.Value; %str2double(handles.currSlice.String);

[mp,mn,me] = fileparts(D.maskInfo.maskPaths{selectedSlice});
sliceMaskName = strcat(mn,me);
maskStruct = load(fullfile(D.guiPrepend, mp, sliceMaskName));
fprintf('Loading masks...\n');

masks = maskStruct.masks;


end