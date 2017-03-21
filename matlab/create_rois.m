function create_rois(D, refMeta, varargin)
   
%  Create ROIs manually with circle-mask, save to struct:

%  Uses a single reference movie for all slices.
%  Create ROIs for each slice from the reference run/movie, then apply this
%  later to each RUN (or file) of that slice.

maskPath = fullfile(D.datastructPath, 'masks');
if ~exist(maskPath, 'dir')
    mkdir(maskPath);
end

nSlices = refMeta.file(1).si.nSlices;

nargin = length(varargin);
if nargin==1
    slicesToUse = varargin{1};
else
    slicesToUse = 1:nSlices;
end

M = struct();

for sidx = slicesToUse %12:2:16 %1:tiff_info.nslices
    refNum = refMeta.file(1).si.motionRefNum; % All files should have some refNum for motion correction.
    
    sliceIdxs = sidx:refMeta.file(refNum).si.nFramesPerVolume:refMeta.file(refNum).si.nTotalFrames;
    
    sliceFns = dir(fullfile(refMeta.file(refNum).si.tiffPath, '*.tif'));
    sliceFns = {sliceFns(:).name}';
    Y = tiffRead(fullfile(refMeta.file(refNum).si.tiffPath, sliceFns{sidx}));
    avgY = mean(Y, 3);

    masks = ROIselect_circle(mat2gray(avgY));
    
    % ------------------------------------
    % TODO:  below is temporary... Line 35 is standard for creating ROIs.
    % Fix ROI select function to store sparse matrices (faster?)...
    %old = load(fullfile(D.datastructPath, 'masks_old', sprintf('masks_Slice%02d_File001.mat', sidx)));
    %masks = old.masks;
    maskcell = cell(size(masks,3),1);
    for roi=1:size(masks,3)
        [i,j,s] = find(masks(:,:,roi));
        [m,n] = size(masks(:,:,roi));
        maskcell{roi} = sparse(i,j,s,m,n); %(:,:,roi);
    end
    % ------------------------------------
    
    M.maskcell = maskcell;
    %M.masks = masks;
    M.slice = sliceFns{sidx};
    M.refPath = refMeta.file(refNum).si.tiffPath; %refMeta.tiffPath;
    M.sliceIdxs = sliceIdxs;
    M.slicePath = fullfile(refMeta.file(refNum).si.tiffPath, sliceFns{sidx});

    % Save reference masks:        
    maskStructName = char(sprintf('masks_Slice%02d_File%03d.mat', sidx, refNum));
    save(fullfile(maskPath, maskStructName), '-struct', 'M', '-v7.3');

end

    
end
