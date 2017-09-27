function mcparams = preprocess_data(mcparams)

% PREPROCESSING:  motion-correction.

% 1. Set parameters for motion-correction.
% 2. If correcting, make standard directories and run correction.
% 3. Re-interleave parsed TIFFs and save in tiff base dir (child of
% functional-dir). Sort parsed TIFFs by Channel-File-Slice.
% 4. Do additional correction for bidirectional scanning (and do
% post-mc-cleanup). [optional]
% 5. Create averaged time-series for all slices using selected correction
% method. Save in standard Channel-File-Slice format.

% Note:  This step also adds fields to mcparams struct that are immutable,
% i.e., standard across all analyses.

% -------------------------------------------------------------------------
% 1.  Use params to do correction on raw TIFFs:
% -------------------------------------------------------------------------
if mcparams.corrected
    
    % Set and create standard 'Corrected' directory in tiff base dir:
    mcparams.corrected_dir = 'Corrected';
    if ~exist(fullfile(mcparams.tiff_dir, mcparams.corrected_dir), 'dir')
        mkdir(fullfile(mcparams.tiff_dir, mcparams.corrected_dir));
    end
    
    % Run specified MC and store method-specific info:
    mcparams.info = do_motion_correction(mcparams);

else
    
    % No MC, so just parse raw tiffs in standard directory in tiff base dir:
    mcparams.parsed_dir = 'Parsed';
    if ~exist(fullfile(mcparams.tiff_dir, mcparams.parsed_dir), 'dir')
        mkdir(fullfile(mcparams.tiff_dir, mcparams.parsed_dir));
    end
    
    % TODO:  add plain raw tiff parsing here.
    % notes:  need to account for flyback-correction check
    
    % Store empty method-specific info struct:
    mcparams.info = struct();
end

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');
fprintf('Completed motion-correction!\n');

% -------------------------------------------------------------------------
% 2.  Clean-up and organize corrected TIFFs into file hierarchy:
% -------------------------------------------------------------------------
% mcparams.split_channels = true;

% TODO:  recreate too-big-TIFF error to make a try-catch statement that
% re-interleaves by default, and otherwise splits the channels if too large

reinterleave_parsed_tiffs(mcparams);

post_mc_cleanup(mcparams);

% -------------------------------------------------------------------------
% 3.  Do additional bidi correction (optional):
% -------------------------------------------------------------------------
% mcparams.bidi_corrected = true;

% TODO:  create MC-metrics from selected MC method, and use some
% metric/thresholding to decide whether to do additional bidi-correction or
% not...  OR, just do this automatically anyway so that all mcparams can be
% usr-specified in set_mc_params()

if mcparams.bidi_corrected
    mcparams.bidi_corrected_dir = 'Corrected_Bidi';
    if ~exist(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir), 'dir')
        mkdir(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir));
    end
end

do_bidi_correction(mcparams);
fprintf('Finished bidi-correction.\n');

% Sort Parsed files into separate directories if needed: 
post_mc_cleanup(mcparams, mcparams.bidi_corrected); 
fprintf('Finished sorting parsed TIFFs.\n')        
save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');


end
