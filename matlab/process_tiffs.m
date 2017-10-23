function [I, A] = process_tiffs(I, A)

% 1. Set parameters for motion-correction.
% 2. If correcting, make standard directories and run correction.
% 3. Re-interleave parsed TIFFs and save in tiff base dir (child of
% functional-dir). Sort parsed TIFFs by Channel-File-Slice.
% 4. Do additional correction for bidirectional scanning (and do
% post-mc-cleanup). [optional]
% 5. Create averaged time-series for all slices using selected correction
% method. Save in standard Channel-File-Slice format.

%if do_preprocessing

% TODO (?):  Run meta-data parsing after
% flyback-correction (py), including SI-meta correction if
% flyback-correction changes the TIFF volumes.

% Load current analysis mcparams:
%curr_mcparams = 

% -------------------------------------------------------------------------
%% 1.  Do Motion-Correction (and/or) Get time-series for each slice:
% -------------------------------------------------------------------------
if I.corrected && new_mc_id && process_raw
    do_motion_correction = true;
elseif I.corrected && (~new_mc_id || ~process_raw)
    found_nchannels = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, '*Channel*'));
    found_nchannels = {found_nchannels(:).name}';
    if length(found_nchannels)>0 && isdir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, found_nchannels{1}))
        found_nslices = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, found_nchannels{1}, '*.tif'));
        found_nslices = {found_nslices(:).name}';
        if length(found_nchannels)==A.nchannels && length(found_nslices)==length(A.nslices)
            fprintf('Found corrected number of deinterleaved TIFFs in Corrected dir.\n');
            user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's')
        end
        if strcmp(user_says_mc, 'Y')
            do_motion_correction = true;
            user_says_delete = input('Deleting old MC folder tree. Press Y to confirm:\n', 's');
            if strcmp(user_says_delete, 'Y')
                rmdir fullfile(mcparams.source_dir, curr_mcparams.dest_dir) s
            end
        elseif strcmp(user_says_mc, 'n')
            do_motion_correction = false;
        end
    else
        found_tiffs = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, '*.tif'));
        found_tiffs = {found_tiffs(:).name}';
        fprintf('Found these TIFFs in Corrected dir - %s:\n', curr_mcparams.dest_dir);
        found_tiffs
        user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's')
        if strcmp(user_says_mc, 'Y')
            do_motion_correction = true;
        elseif strcmp(user_says_mc, 'n')
            do_motion_correction = false;
        end
    end
else
    % Just parse raw tiffs:
    do_motion_correction = false;
end

% CHECK anyway:
if do_motion_correction
    % This is redundant to above, but double-check temporarily: 
    found_nfiles = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, '*.tif'));
    found_nfiles = {found_nfiles(:).name}';
    if length(found_nfiles)==A.ntiffs
        fprintf('Found correct number of interleaved TIFFs in Corrected dir: %s\n', curr_mcparams.dest_dir);
        user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's');
    end
    if strcmp(user_says_mc, 'Y')
        do_motion_correction = true;
        user_says_delete = input('Deleting old MC folder tree. Press Y to confirm:\n', 's');
        if strcmp(user_says_delete, 'Y')
            rmdir fullfile(mcparams.source_dir, curr_mcparams.dest_dir) s
        end
    elseif strcmp(user_says_mc, 'n')
        do_motion_correction = false;
    end
end


if do_motion_correction
    % Do motion-correction on "raw" tiffs in ./DATA, save slice tiffs to curr_mcparams.dest_dir
    % Rename deinterleaved-corrected TIFFs to ./DATA/<curr_mcparams.dest_dir>_slices in next step.
    % Interleave and save interleaved-corrected TIFFs to ./DATA/<curr_mcparams.dest_dir. in next step.
    curr_mcparams = motion_correct_data(curr_mcparams);
    fprintf('Completed motion-correction!\n');
else
    % Just parse "raw" tiffs in ./DATA to ./DATA/Raw_slices, then move original tiffs to ./DATA/Raw
    fprintf('Not doing motion-correction...\n');
    curr_mcparams = create_deinterleaved_tiffs(A, curr_mcparams);
end

fprintf('Finished motion-correcting TIFF data.\n');

% -------------------------------------------------------------------------
%% 3.  Clean-up and organize corrected TIFFs into file hierarchy:
% -------------------------------------------------------------------------
% mcparams.split_channels = true;

% TODO:  recreate too-big-TIFF error to make a try-catch statement that
% re-interleaves by default, and otherwise splits the channels if too large

% Reinterleave slices tiffs into full Files (only if ran motion-correction):
% Don't need to reinterleave and sort TIFFs if just parsing raw, since that is done in create_deinterleaved_tiffs()

% PYTHON equivalent faster?: 
deinterleaved_tiff_dir = fullfile(curr_mcparams.source_dir, sprintf('%s_slices', curr_mcparams.dest_dir));
if I.corrected && process_raw
    if ~isdir(deinterleaved_tiff_dir)
        movefile(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir), deinterleaved_tiff_dir);
    end
    reinterleaved_tiff_dir = fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir);
    reinterleave_tiffs(A, deinterleaved_tiff_dir, reinterleaved_tiff_dir, curr_mcparams.split_channels);
end

% Sort parsed slices by Channel-File:
post_mc_cleanup(deinterleaved_tiff_dir, A);

fprintf('Got corrected, interleaved tiffs. Done with post-MC cleanup.\n');

% -------------------------------------------------------------------------
%% 4.  Do additional bidi correction (optional)
% -------------------------------------------------------------------------
% mcparams.bidi_corrected = true;

if curr_mcparams.bidi_corrected
    if curr_mcparams.corrected
        if process_raw
            bidi_source = sprintf('%s_%s', 'Corrected', I.mc_id);
        else
            bidi_source = curr_mcparams.dest_dir;
        end
    else
        bidi_source = 'Raw';
    end
    curr_mcparams = do_bidi_correction(bidi_source, curr_mcparams, A);
end

% NOTE: bidi function updates mcparams.dest_dir 
deinterleaved_tiff_dir = fullfile(curr_mcparams.source_dir, sprintf('%s_slices', curr_mcparams.dest_dir));

% Sort parsed slices by Channel-File:
post_mc_cleanup(deinterleaved_tiff_dir, A);

fprintf('Finished BIDI correction step.\n')

% -------------------------------------------------------------------------
%% 5.  Create averaged slices from desired source:
% -------------------------------------------------------------------------

source_tiff_basepath = fullfile(curr_mcparams.source_dir, sprintf('%s_slices', I.average_source));
dest_tiff_basepath = fullfile(curr_mcparams.source_dir, sprintf('Averaged_Slices_%s', I.average_source));

create_averaged_slices(source_tiff_basepath, dest_tiff_basepath, A);
fprintf('Finished creating AVERAGED slices from dir %s\n', I.average_source);

% Update mcparams struct and reference struct:
curr_mcparams.averaged_slices_dir = dest_tiff_basepath;

% Also save MCPARAMS as json:
[ddir, fname, ext] = fileparts(A.mcparams_path);
mcparams_json = fullfile(ddir, strcat(fname, '.json'));
%savejson('', mcparams, mcparams_json);    
% TODO:  this does not save properly of n-fields (mcparam ids) > 2...

save(path_to_reference, '-struct', 'A', '-append')

% Also save json:
savejson('', A, path_to_reference_json);

fprintf('Finished preprocessing data.\n');

% -------------------------------------------------------------------------
%% 6.  Create updated simeta struct, if doesn't exist
% -------------------------------------------------------------------------
if ~exist(fullfile(curr_mcparams.source_dir, sprintf('%s.mat', A.base_filename)), 'file')
    raw_simeta = load(A.raw_simeta_path);
    metaDataSI = {};
    for fi=1:A.ntiffs
        currfile = sprintf('File%03d', fi);
        tmpSI = raw_simeta.(currfile);
        if curr_mcparams.flyback_corrected
            curr_simeta = adjust_si_metadata(tmpSI, mov_size);
        else
            curr_simeta = tmpSI;
        end
        metaDataSI{fi} = curr_simeta;
    end
    eval([A.base_filename '=struct();'])
    eval([(A.base_filename) '.metaDataSI = metaDataSI'])
    mfilename = sprintf('%s', A.base_filename);
    save(fullfile(curr_mcparams.source_dir, mfilename), mfilename);
    fprintf('Created updated SI metadata for flyback-corrected TIFFs to:\n')
    fprintf('%s\n', fullfile(curr_mcparams.source_dir, mfilename));
end

mcparams.(mc_id) = curr_mcparams;
save(A.mcparams_path, '-struct', 'mcparams');

end
