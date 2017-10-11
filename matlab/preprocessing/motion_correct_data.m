function [A, mcparams]= motion_correct_data(A, mcparams)

simeta = load(A.raw_simeta_path);

% -------------------------------------------------------------------------
% 1.  Use params to do correction on raw TIFFs:
% -------------------------------------------------------------------------
    
% Set and create standard 'Corrected' directory in tiff base dir:
mcparams.corrected_dir = 'Corrected';
if ~exist(fullfile(mcparams.tiff_dir, mcparams.corrected_dir), 'dir')
    mkdir(fullfile(mcparams.tiff_dir, mcparams.corrected_dir));
end

% Run specified MC and store method-specific info:
mcparams.info = do_motion_correction(mcparams)
   
% TODO:  add plain raw tiff parsing here.
% notes:  need to account for flyback-correction check


save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');
fprintf('Completed motion-correction!\n');

% Add base filename if missing from ref struct:
% For now, this is specific to Acquisition2P, since this uses the base filename to name field for acq obj.
if ~isfield(A, 'base_filename') || ~strcmp(A.base_filename, mcparams.info.acquisition_name)
    A.base_filename = mcparams.info.acquisition_name;
end

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');
%save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');


end