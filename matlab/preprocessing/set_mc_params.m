%% Set preprocessing parameters:

% 1.  Acquisition info:
% -------------------------------------------------------------------------
source = '/nas/volume1/2photon/projects/gratings_phaseMod';
session = '20170901_CE054_zoom3x';
acquisition = 'functional_test';

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
    acquisition = fullfile(acquisition, 'DATA');
end

acquisition_dir = fullfile(source, session, acquisition);
mcparams.acquisition_dir = acquisition_dir;

