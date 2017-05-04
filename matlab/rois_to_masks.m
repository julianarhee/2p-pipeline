function rois_to_masks(D)
   
%  ROIs are created automatically in python (saved to .mat file).
%  This assumes that ROI segmentation was done on some averaged volume
%  (tested with TeFo data...), and creates a set of masks from a list of
%  ROIs for each slice.
%  ROIs :  list of [y, x, r]

%  Create ROIs for each slice from the reference (i.e., averaged) volume,
%  and save to standard mask struct (sparse mats in cell array, where
%  length(maskcell)=nROIs for a given slice.

%  Standard format :  masks_Slice00x.mat for each slice, containing F
%  fields "file" for each file in current acquisition.

maskPath = fullfile(D.datastructPath, 'masks');
if ~exist(maskPath, 'dir')
    mkdir(maskPath);
end

rois = load(fullfile(D.maskInfo.mapSource, D.maskInfo.roiPath)); % ROI keys are slices with 1-indexing

slicesToUse = D.slices;

maskSlicePaths = dir(fullfile(D.maskInfo.mapSlicePaths, '*.tif'));
maskSlicePaths = {maskSlicePaths(:).name}';

M = struct();

for sidx = slicesToUse %12:2:16 %1:tiff_info.nslices
    
    currslice_name = sprintf('slice%i', sidx);
    switch D.maskInfo.blobType
        case 'lagrange'
            currRois = rois.(currslice_name).LoG;
        case 'difference'
            currRois = rois.(currslice_name).DoG;
    end
    
    %sliceIdxs = sidx:refMeta.file(refNum).si.nFramesPerVolume:refMeta.file(refNum).si.nTotalFrames;
    mapSlicePath = fullfile(D.maskInfo.mapSlicePaths, sprintf('%i.tif', sidx-1));
    mapSliceImg = tiffRead(mapSlicePath);
    
    centers = [double(currRois(:,2)), double(currRois(:,1))];
    radii = double(currRois(:,3))-0.3;
    
    [dim1, dim2] = size(mapSliceImg);
    
    [colsInImage rowsInImage] = meshgrid(1:dim1, 1:dim2);
        
    circfunc = @(r) sqrt((rowsInImage - centers(r,2)).^2 + (colsInImage - centers(r,1)).^2) < radii(r).^2;
    
    nrois = length(currRois);
    masks = arrayfun(@(roi) circfunc(roi), 1:nrois, 'UniformOutput', false);
    masks = cat(3,masks{1:end});
%     RGBimg = zeros(dim1, dim2, 3);
%     RGBimg(:,:,2) = mat2gray(mapSliceImg);
%     for r=1:nrois
%         RGBimg(:,:,3) = RGBimg(:,:,3) + 0.5*masks{r};
%     end
%     figure(); imagesc(RGBimg)
%     


    %masks = ROIselect_circle(mat2gray(avgY));
    maskcell = arrayfun(@(roi) makeSparseMasks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
    % ------------------------------------
    % TODO:  below is temporary... Line 35 is standard for creating ROIs.
    % Fix ROI select function to store sparse matrices (faster?)...

%     old = load(fullfile(D.datastructPath, 'masks_old', sprintf('masks_Slice%02d_File001.mat', sidx)));
%     
%     if ~isfield(old, 'maskcell')
%         masks = old.masks;
%         tic()
%         maskcell = arrayfun(@(roi) makeSparseMasks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
%         toc()
%     else
%         maskcell = old.maskcell;
%     end
    
%     maskcell = cell(size(masks,3),1);
%     tic()
%     for roi=1:size(masks,3)
%         [i,j,s] = find(masks(:,:,roi));
%         [m,n] = size(masks(:,:,roi));
%         maskcell{roi} = sparse(i,j,s,m,n); %(:,:,roi);
%     end
%     toc()
    maskcell = cellfun(@logical, maskcell, 'UniformOutput', false);
    % ------------------------------------
    
    M.maskcell = maskcell;
    %M.masks = masks;
    M.sliceImg = mapSliceImg;
    M.slicePath = mapSlicePath;
    
    %M.refPath = refMeta.file(refNum).si.tiffPath; %refMeta.tiffPath;

    % Save reference masks:        
    pathprts = strsplit(D.maskInfo.mapSource, '/');
    
    maskStructName = char(sprintf('masks_%s_Slice%02d.mat', pathprts{end}, sidx));
    save(fullfile(maskPath, maskStructName), '-struct', 'M', '-v7.3');

end

    
end
