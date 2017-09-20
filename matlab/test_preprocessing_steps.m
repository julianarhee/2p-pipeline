% PREPROCESSING.

%% Run script to populate MC params:
set_mc_params;

%% Do motion-correction:
mcparams = do_motion_correction(mcparams);

%% Clean-up and organize corrected TIFFs into file hierarchy:

split_channels = false;
post_mc_cleanup(mcparams, split_channels);


%%
