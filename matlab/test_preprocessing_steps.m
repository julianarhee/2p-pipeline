%% PREPROCESSING.


%% Do motion correction:

% 1.  Set MC parameters:
set_mc_params;

% 2.  Use params to do correction on raw TIFFs:
mcparams = do_motion_correction(mcparams);
save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams');
fprintf('Completed motion-correction!\n');

% 3.  Clean-up and organize corrected TIFFs into file hierarchy:
mcparams.split_channels = true;
reinterleave_parsed_tiffs(mcparams);

post_mc_cleanup(mcparams);

%% Do additional bidi correction (optional):

correct_bidi = true;
mcparams.correct_bidi = correct_bidi;

mcparams.bidi_corrected_dir = fullfile(mcparams.tiff_dir, 'Corrected_Bidi');
if ~exist(mcparams.bidi_corrected_dir, 'dir')
    mkdir(mcparams.bidi_corrected_dir);
end
do_bidi_correction(mcparams);
fprintf('Finished bidi-correction.\n');

% Sort Parsed files into separate directories if needed: 
bidi=mcparams.correct_bidi;
post_mc_cleanup(mcparams, bidi); 
fprintf('Finished sorting parsed TIFFs.\n')        
save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');

%% Create and save average slices:

mcparams.averaged_slices_dir = fullfile(mcparams.tiff_dir, 'Averaged_Slices');
if ~exist(mcparams.averaged_slices_dir, 'dir')
    mkdir(mcparams.averaged_slices_dir);
end
create_averaged_slices(mcparams);
fprintf('Finished creating average slices.\n');

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');
