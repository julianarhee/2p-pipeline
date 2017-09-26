%% Set preprocessing parameters:

% 1. Specify Motion Correction params:
% -------------------------------------------------------------------------
% run_multi_acquisitions = false;
mcparams.corrected = true;
mcparams.method = 'Acquisition2P'; % ['Acqusition2P', 'NoRMCorre']
mcparams.crossref = false;
mcparams.processed_flyback = true;

mcparams.ref_channel = 1;
mcparams.ref_file = 3;
mcparams.algorithm = @lucasKanade_plus_nonrigid; % @withinFile_withinFrame_lucasKanade;
% -------------------------------------------------------------------------

mcparams.tiff_dir = A.data_dir;

