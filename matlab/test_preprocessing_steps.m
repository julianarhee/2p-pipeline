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

save(strcat(mcparams.acquisition_dir, 'mcparams.mat'), 'mcparams');

%% Clean-up and organize corrected TIFFs into file hierarchy:

%split_channels = false;
%post_mc_cleanup(mcparams, split_channels);

%% Do additional bidi correction (optional):

correct_bidi = true;
mcparams.correct_bidi = correct_bidi;
mcparams.bidi_corrected_dir = fullfile(mcparams.acquisition_dir, 'Corrected_Bidi');


%% Create and save average slices:


