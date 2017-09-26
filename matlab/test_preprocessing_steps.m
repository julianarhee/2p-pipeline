%% PREPROCESSING:  motion-correction.

% 1. Set parameters for motion-correction.
% 2. If correcting, make standard directories and run correction.
% 3. Re-interleave parsed TIFFs and save in tiff base dir (child of
% functional-dir). Sort parsed TIFFs by Channel-File-Slice.
% 4. Do additional correction for bidirectional scanning (and do
% post-mc-cleanup). [optional]
% 5. Create averaged time-series for all slices using selected correction
% method. Save in standard Channel-File-Slice format.

%% 1.  Set MC parameters:
set_mc_params;

%% 2.  Use params to do correction on raw TIFFs:
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

save(A.mcparams_path, 'mcparams');
fprintf('Completed motion-correction!\n');

%% 3.  Clean-up and organize corrected TIFFs into file hierarchy:

mcparams.split_channels = true;
reinterleave_parsed_tiffs(mcparams);

post_mc_cleanup(mcparams);

%% 4.  Do additional bidi correction (optional):

mcparams.bidi_corrected = true;
if mcparams.bidi_corrected
    mcparams.bidi_corrected_dir = 'Corrected_Bidi';
    if ~exist(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir), 'dir')
        mkdir(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir));
    end
end

do_bidi_correction(mcparams);
fprintf('Finished bidi-correction.\n');

% Sort Parsed files into separate directories if needed: 
bidi=mcparams.bidi_corrected;
post_mc_cleanup(mcparams, bidi); 
fprintf('Finished sorting parsed TIFFs.\n')        
save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');

%% 5.  Create and save average slices:

mcparams.averaged_slices_dir = 'Averaged_Slices';
if ~exist(fullfile(mcparams.tiff_dir, mcparams.averaged_slices_dir), 'dir')
    mkdir(fullfile(mcparams.tiff_dir, mcparams.averaged_slices_dir));
end
create_averaged_slices(mcparams);
fprintf('Finished creating average slices.\n');

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');
