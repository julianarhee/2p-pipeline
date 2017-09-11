% set params for motion-correction, if using Acquisition2P_class
run_multi_acquisitions = false;

crossref = false;
processed = true;

source = '/nas/volume1/2photon/projects/retino_bar';
session = '20170902_CE054';
run = 'functional_zoom3x_run1';
if processed
    fprintf('Motion correcting processed tiffs.\n');
    run = fullfile(run, 'DATA');
end

mc_ref_channel = 1;
mc_ref_movie = 1;

acquisition_dir = fullfile(source, session, run);

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

% TODO: set case statements to choose either Acquisition2P or NoRMCorre here.
motion_correct_acquisitions;
fprintf('Completed Acqusition2P motion-correction!\n');




