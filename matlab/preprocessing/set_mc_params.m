%% Set preprocessing parameters:

% 1.  Acquisition info:
% -------------------------------------------------------------------------
source = '/nas/volume1/2photon/projects/retino_bar';
session = '20170902_CE054';
run = 'functional_zoom3x_run1';

% 2.  Motion Correction info:
% -------------------------------------------------------------------------
% run_multi_acquisitions = false;
mcparams.source = 'Acquisition2P'; % ['Acqusition2P', 'NoRMCorre']
mcparams.crossref = false;
mcparams.processed = true;

mcparams.ref_channel = 1;
mcparams.ref_movie = 1;
mcparams.algorithm = @lucasKanade_plus_nonrigid; % @withinFile_withinFrame_lucasKanade;
% -------------------------------------------------------------------------

if mcparams.processed
    fprintf('Motion correcting processed tiffs.\n');
    run = fullfile(run, 'DATA');
end

acquisition_dir = fullfile(source, session, run);
mcparams.acquisition_dir = acquisition_dir;

