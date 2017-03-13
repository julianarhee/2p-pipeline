function create_rois(analysis_dir, acquisition_name, refMeta, slice_fns, varargin)
   
%  Create ROIs manually with circle-mask, save to struct:

%  Uses a single reference movie for all slices.
%  Create ROIs for each slice from the reference run/movie, then apply this
%  later to each RUN (or file) of that slice.

nargin = length(varargin);
if nargin==1
    slices_to_use = varargin{1};
else
    slices_to_use = 1:length(slice_fns);
end

M = struct();
refNum = refMeta.tiff_fidx;


for sidx = slices_to_use %12:2:16 %1:tiff_info.nslices
    slice_indices = sidx:refMeta.nframes_per_volume:refMeta.ntotal_frames;

    Y = tiffRead(fullfile(refMeta.tiff_path, slice_fns{sidx}));
    avgY = mean(Y, 3);

    masks = ROIselect_circle(mat2gray(avgY));

    M.masks = masks;
    M.slice = slice_fns{sidx};
    M.refPath = refMeta.tiff_path;
    M.slice_indices = slice_indices;

    % Save reference masks:        
    struct_save_path = fullfile(analysis_dir, 'masks');
    if ~exist(struct_save_path, 'dir')
        mkdir(struct_save_path);
    end

    mask_struct_fn = char(sprintf('masks_Slice%02d_File%03d.mat', sidx, refNum));
    save(fullfile(struct_save_path, mask_struct_fn), 'M', '-struct', '-v7.3');

end

    
end
