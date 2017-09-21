% PREPROCESSING.
%% Clear all and make sure paths set

clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Run script to populate MC params:
set_mc_params;

%% Do motion-correction:
mcparams = do_motion_correction(mcparams);

save(fullfile(mcparams.acquisition_dir, 'mcparams.mat'), 'mcparams');

%% Clean-up and organize corrected TIFFs into file hierarchy:

mcparams.split_channels = true;
post_mc_cleanup(mcparams);

%% Do additional bidi correction (optional):

correct_bidi = true;
mcparams.correct_bidi = correct_bidi;

mcparams.bidi_corrected_dir = fullfile(mcparams.acquisition_dir, 'Corrected_Bidi');
if ~exist(mcparams.bidi_corrected_dir, 'dir')
    mkdir(mcparams.bidi_corrected_dir);
end
do_bidi_correction(mcparams);


%% Create and save average slices:

mcparams.averaged_slices_dir = fullfile(mcparams.acquisition_dir, 'Averaged_Slices');
if ~exist(mcparams.averaged_slices_dir, 'dir')
    mkdir(mcparams.averaged_slices_dir);
end
create_averaged_slices(mcparams);


