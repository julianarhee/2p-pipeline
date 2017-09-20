%% Clear all and make sure paths set

clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Set preprocessing parameters:

% 1.  Acquisition info:
% -------------------------------------------------------------------------
source = '/nas/volume1/2photon/projects/retino_bar';
session = '20170902_CE054';
run = 'functional_zoom3x_run1';
if processed
    fprintf('Motion correcting processed tiffs.\n');
    run = fullfile(run, 'DATA');
end

acquisition_dir = fullfile(source, session, run);

% 2.  Motion Correction info:
% -------------------------------------------------------------------------
% run_multi_acquisitions = false;
mcparams.acqusition_dir = acquisition_dir;
mcparams.source = 'Acquisition2P'; % ['Acqusition2P', 'NoRMCorre']
mcparams.crossref = false;
mcparams.processed = true;

mcparams.ref_channel = 1;
mcparams.ref_movie = 1;
mcparams.algorithm = @lucasKanade_plus_nonrigid; % @withinFile_withinFrame_lucasKanade;





