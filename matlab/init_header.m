% Put in .gitignore at top of repo. Edit to run specific acquisitions.

useGUI = false;
get_rois_and_traces = false %true %false;
do_preprocessing = true %false %true;

if ~useGUI
    % Set info manually:
    source = '/nas/volume1/2photon/projects';
    experiment = 'gratings_phaseMod';
    session = '20171005_CE059';
    acquisition = 'FOV1_zoom3x'; %'FOV1_zoom3x';
    tiff_source = 'functional'; %'functional_subset';
    acquisition_base_dir = fullfile(source, experiment, session, acquisition);
    curr_tiff_dir = fullfile(acquisition_base_dir, tiff_source);
end
