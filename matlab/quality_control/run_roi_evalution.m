
clear all

% Set info manually:
source = '/nas/volume1/2photon/projects';
experiment = 'scenes'; %'gratings_phaseMod'; %'scenes';
session = '20171003_JW016'; %'20171009_CE059'; %'20171003_JW016';
acquisition = 'FOV1'; %'FOV1_zoom3x';
tiff_source = 'functional'; %'functional_subset';
acquisition_base_dir = fullfile(source, experiment, session, acquisition);
curr_tiff_dir = fullfile(acquisition_base_dir, tiff_source);

%put expected info into options structure
options = struct;
options.acquisition_base_dir = acquisition_base_dir;
options.tiff_source = tiff_source;
options.data_dir = fullfile(acquisition_base_dir, tiff_source, 'DATA');
options.get_trial_corr = 0;
options.get_pixel_corr = 1;
options.get_file_corr = 1;
options.mcparams_id = sprintf('mcparams%02d',1);
options.analysis_id = sprintf('analysis%02d',1);
options.roi_folder = 'blobs_DoG';%name of ROI subfolder
options.sourceslices = 5:5:41;

%run function
evaluate_rois(options)