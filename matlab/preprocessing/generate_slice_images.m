function mcparams = generate_slice_images(mcparams)

% 5.  Create and save average slices:

mcparams.averaged_slices_dir = 'Averaged_Slices';
if ~exist(fullfile(mcparams.tiff_dir, mcparams.averaged_slices_dir), 'dir')
    mkdir(fullfile(mcparams.tiff_dir, mcparams.averaged_slices_dir));
end
create_averaged_slices(mcparams);
fprintf('Finished creating average slices.\n');

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');

end