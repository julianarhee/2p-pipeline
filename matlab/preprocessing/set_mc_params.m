%% Set preprocessing parameters:

% 1. Specify Motion Correction params:
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
    acquisition_file_dir = fullfile(A.acquisition, 'DATA');
else
    acquisition_file_dir = A.acquisition;
end

tiff_dir = fullfile(A.source, A.session, acquisition_file_dir);
mcparams.tiff_dir = tiff_dir;

