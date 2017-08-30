function create_rois(D, refMeta)
   
%  Create ROIs manually with circle-mask, save to struct:

%  Uses a single reference movie for all slices.
%  Create ROIs for each slice from the reference run/movie, then apply this
%  later to each RUN (or file) of that slice.

maskPath = fullfile(D.datastructPath, 'masks');
if ~exist(maskPath, 'dir')
    mkdir(maskPath);
end

%nSlices = refMeta.file(1).si.nSlices;
%meta = load(fullfile(D.sourceDir, D.metaPath));

slicesToUse = D.slices;

M = struct();
pathToMasks = {};

for sidx = slicesToUse %12:2:16 %1:tiff_info.nslices
    
    if strcmp(D.roiType, 'manual3Drois')
        refNum = D.localRefNum;
    else
        refNum = D.refRun; %refMeta.file(1).si.motionRefNum; % All files should have some refNum for motion correction.
    end
    
    %sliceIdxs = sidx:refMeta.file(refNum).si.nFramesPerVolume:refMeta.file(refNum).si.nTotalFrames;

    sliceFns = dir(fullfile(refMeta.file(refNum).si.tiffPath, '*.tif'));
    sliceFns = {sliceFns(:).name}';
    Y = tiffRead(fullfile(refMeta.file(refNum).si.tiffPath, sliceFns{sidx}));
    avgY = mean(Y, 3);

    masks = ROIselect_circle(mat2gray(avgY));
    maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
    % ------------------------------------
    % TODO:  below is temporary... Line 35 is standard for creating ROIs.
    % Fix ROI select function to store sparse matrices (faster?)...

%     old = load(fullfile(D.datastructPath, 'masks_old', sprintf('masks_Slice%02d_File001.mat', sidx)));
%     
%     if ~isfield(old, 'maskcell')
%         masks = old.masks;
%         tic()
%         maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
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
    M.slice = sliceFns{sidx};
    M.refPath = refMeta.file(refNum).si.tiffPath; %refMeta.tiffPath;
    %M.sliceIdxs = sliceIdxs;
    M.slicePath = fullfile(refMeta.file(refNum).si.tiffPath, sliceFns{sidx});

    % Save reference masks:        
    maskStructName = char(sprintf('masks_%s_Slice%02d_File%03d.mat', D.experiment, sidx, refNum));
    save(fullfile(maskPath, maskStructName), '-struct', 'M', '-v7.3');
    pathToMasks{end+1} = maskStructName;

end

    
end
