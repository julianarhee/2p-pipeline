function [maskcell, roiIdxs] = getCurrentSliceMasks(handles,D)

selectedSlice = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedFile = handles.runMenu.Value;

if selectedSlice > length(D.slices)
    selectedSlice = length(D.slices);
end

if strcmp(D.roiType, 'pixels')
    
    meta = load(D.metaPath);
    d1 = meta.file(1).si.frameWidth;
    d2 = meta.file(1).si.linesPerFrame;
    nrois = d1*d2;
    
    %maskcell = arrayfun(@(i) ind2sub([d1, d2], i), 1:nrois, 'UniformOutput', 0);
    maskcell = cell(1,nrois);
    for i=1:nrois
        maskcell{i} = [0, 0];
        [maskcell{i}(1), maskcell{i}(2)] = ind2sub([d1, d2], i);
    end
    
    roiIdxs = 1:length(maskcell);
    
else

    if strfind(D.roiType, '3D')
        [mp,mn,me] = fileparts(D.maskPaths{selectedSlice});
    else
        [mp,mn,me] = fileparts(D.maskInfo.maskPaths{selectedSlice});
    end
    sliceMaskName = strcat(mn,me);
    maskStruct = load(fullfile(D.guiPrepend, mp, sliceMaskName));
    fprintf('Loading masks...\n');

    if isfield(maskStruct, 'file')
        maskcell = maskStruct.file(selectedFile).maskcell; % masks created with create_rois.m
%         while isempty(maskcell)
% %             for fidx=1:length(maskStruct.file)
% %                 maskcell = maskStruct.file(fidx).maskcell;
% %                 handles.runMenu.Value = fidx;
% %             end
%             for sidx=1:length(D.slices)
%                 tmpmasks = load(D.maskPaths{sidx});
%                 maskcell = tmpmasks.file(selectedFile).maskcell;
%             end
%             selectedSlice = sidx;
%             handles.currSlice.Value = selectedSlice;
%                 
%         end

        if isempty(maskcell)
            roiIdxs = 1;
            maskcell = {0};
        else
            if strfind(D.roiType, '3D') 
                if isfield(maskStruct.file(handles.runMenu.Value), 'roi3Didxs')
                    roiIdxs = maskStruct.file(handles.runMenu.Value).roi3Didxs;
                else
                   roiIdxs = maskStruct.file(handles.runMenu.Value).roiIDs;
                end
            else
                roiIdxs = 1:length(maskcell);
            end
        end

    else
        maskcell = maskStruct.maskcell;
        roiIdxs = 1:length(maskcell);

    end

end

end