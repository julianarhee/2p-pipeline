%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Select TIFF dirs for current analysis

noUI = true;
get_rois_and_traces = false;
do_preprocessing = true;

if noUI

    % Set info manually:
    source = '/nas/volume1/2photon/projects';
    experiment = 'scenes'; %'gratings_phaseMod';
    session = '20171003_JW016'; %'20170927_CE059';
    acquisition = 'FOV1'; %'FOV1_zoom3x';
    tiff_source = 'functional'; %'functional_subset';
    acquisition_base_dir = fullfile(source, experiment, session, acquisition);
    curr_tiff_dir = fullfile(acquisition_base_dir, tiff_source);
else
    default_root = '/nas/volume1/2photon/projects';                            % DIR containing all experimental data
    tiff_dirs = uipickfiles('FilterSpec', default_root);                       % Returns cell-array of full paths to selected folders containing TIFFs to be processed
    
    % For now, just do 1 dir, but later can iterate over each TIFF dir
    % selected:
    curr_tiff_dir = tiff_dirs{1};

end
    
%% Get PY-created Acquisition Struct. Add paths/params for MAT steps:

% Iterate through selected tiff-folders to build paths:
% TODO:  do similar selection step for PYTHON (Step1 preprocessing)

[tiff_parent, tiff_source, ~] = fileparts(curr_tiff_dir);
[acq_parent, acquisition, ~] = fileparts(tiff_parent);
[sess_parent, session, ~] = fileparts(acq_parent);
[source, experiment, ~] = fileparts(sess_parent);


% Build acq path and get reference struct:
% ----------------------------------------
acquisition_base_dir = fullfile(source, experiment, session, acquisition)
path_to_reference = fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source))
A = load(path_to_reference);


%% PREPROCESSING:  motion-correction.

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

if do_preprocessing
    A.acquisition_base_dir = acquisition_base_dir;
    A.data_dir = fullfile(A.acquisition_base_dir, A.functional, 'DATA');

    % -------------------------------------------------------------------------
    %% 1.  Set MC params
    % -------------------------------------------------------------------------

    A.use_bidi_corrected = false;                              % Extra correction for bidi-scanning for extracting ROIs/traces (set mcparams.bidi_corrected=true)
    A.signal_channel = 1;                                      % If multi-channel, Ch index for extracting activity traces

    % Names = [
    %     'corrected          '       % corrected or raw (T/F)
    %     'method             '       % Source for doing correction. Can be custom. ['Acqusition2P', 'NoRMCorre']
    %     'flyback_corrected  '       % True if did correct_flyback.py 
    %     'ref_channel        '       % Ch to use as reference for correction
    %     'ref_file           '       % File index (of numerically-ordered TIFFs) to use as reference
    %     'algorithm          '       % Depends on 'method': Acq_2P [@lucasKanade_plus_nonrigid, @withinFile_withinFrame_lucasKanade], NoRMCorre ['rigid', 'nonrigid']
    %     'split_channels     '       % *MC methods parse corrected-tiffs by Channel-File-Slice (Acq2P does this already). Last step interleaves parsed tiffs, but sometimes they are too big for Matlab
    %     'bidi_corrected     '       % *For faster scanning, SI option for bidirectional-scanning is True -- sometimes need extra scan-phase correction for this
    %     ];

    mcparams = set_mc_params(...
        'corrected', 'true',...
        'method', 'Acquisition2P',...
        'flyback_corrected', true,...
        'ref_channel', 1,...
        'ref_file', 2,...
        'algorithm', @lucasKanade_plus_nonrigid,...
        'split_channels', false,...
        'bidi_corrected', false,...
        'tiff_dir', A.data_dir,...
        'nchannels', A.nchannels);              
    % ----------------------------------------------------------------------------------
    A.corrected = mcparams.corrected; 
    A.mcparams_path = fullfile(A.data_dir, 'mcparams.mat');    % Standard path to mcparams struct (don't change)
    save(A.mcparams_path, 'mcparams');
    save(path_to_reference, '-struct', 'A', '-append');

    % TODO (?):  Run meta-data parsing after
    % flyback-correction (py), including SI-meta correction if
    % flyback-correction changes the TIFF volumes.

    % -------------------------------------------------------------------------
    %% 2.  Do Motion-Correction (and/or) Get Slice t-series:
    % -------------------------------------------------------------------------

    if A.corrected
        %[A, mcparams] = preprocess_data(A, mcparams);                      % include mcparams as output since paths are updated during preprocessing (path(s) to Corrected/Parsed files)
       [A, mcparams] = motion_correct_data(A, mcparams);
    else
        % Just parse raw tiffs:
        [A, mcparams] = create_deinterleaved_tiffs(A, mcparams);
    end

    % -------------------------------------------------------------------------
    %% 3.  Clean-up and organize corrected TIFFs into file hierarchy:
    % -------------------------------------------------------------------------
    % mcparams.split_channels = true;
        
    % TODO:  recreate too-big-TIFF error to make a try-catch statement that
    % re-interleaves by default, and otherwise splits the channels if too large
    
    % PYTHON equivalent faster: 
    if A.corrected
        deinterleaved_source = mcparams.corrected_dir;
    else
        deinterleaved_source = mcparams.parsed_dir;
    end
    deinterleaved_tiff_dir = fullfile(A.data_dir, deinterleaved_source);
    reinterleave_tiffs(A, deinterleaved_tiff_dir, A.data_dir, mcparams.split_channels);

    % Sort parsed slices by Channel-File:
    path_to_cleanup = fullfile(mcparams.tiff_dir, mcparams.corrected_dir);
    post_mc_cleanup(path_to_cleanup, A);


    % -------------------------------------------------------------------------
    %% 4.  Do additional bidi correction (optional)
    % -------------------------------------------------------------------------
    % mcparams.bidi_corrected = true;

    if A.use_bidi_corrected
        [A, mcparams] = do_bidi_correction(A, mcparams);
    end
    
    % Sort bidi-corrected:
    if isfield(mcparams, 'bidi_corrected_dir')
        path_to_cleanup = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
        post_mc_cleanup(path_to_cleanup, A);     
    end

    % -------------------------------------------------------------------------
    %% 5.  Create averaged slices from desired source:
    % -------------------------------------------------------------------------

    if A.corrected
        if A.use_bidi_corrected                                % TODO: may want to have intermediate step to evaluate first MC step...
            A.source_to_average = mcparams.bidi_corrected_dir;
        else
            A.source_to_average = mcparams.corrected_dir;
        end
    else
        A.source_to_average = mcparams.parsed_dir;               % if no correction is done in preprocessing step above, still parse tiffs by slice to get t-series
    end

    source_tiff_basepath = fullfile(mcparams.tiff_dir, A.source_to_average);
    dest_tiff_basepath = fullfile(mcparams.tiff_dir, sprintf('Averaged_Slices_%s', A.source_to_average));
    mcparams.averaged_slices_dir = dest_tiff_basepath;
    save(A.mcparams_path, 'mcparams', '-append');
 
    create_averaged_slices(source_tiff_basepath, dest_tiff_basepath, A);
    save(path_to_reference, '-struct', 'A', '-append')

    fprintf('Finished preprocessing data.\n');

end


%% Specify ROI param struct path:
if get_rois_and_traces
    A.roi_method = 'pyblob2D';
    A.roi_id = 'blobs_DoG';

    A.roiparams_path = fullfile(A.acquisition_base_dir, 'ROIs', A.roi_id, 'roiparams.mat');

    %% GET ROIS.


    %% Specify Traces param struct path:

    A.trace_id = 'blobs_DoG';
    A.trace_dir = fullfile(A.acquisition_base_dir, 'Traces', A.trace_id);
    if ~exist(A.trace_dir, 'dir')
        mkdir(A.trace_dir)
    end

    %% Get traces
    extract_traces(A);
    fprintf('Extracted raw traces.\n')

    %% GET metadata for SI tiffs:

    si = get_scan_info(A)
    save(fullfile(A.data_dir, 'simeta.mat'), '-struct', 'si');
    A.simeta_path = fullfile(A.data_dir, 'simeta.mat');

    %% Process traces

    % For retino-movie:
    % targetFreq = meta.file(1).mw.targetFreq;
    % winUnit = (1/targetFreq);
    % crop = meta.file(1).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
    % nWinUnits = 3;

    % For PSTH:
    win_unit = 3; 
    num_units = 3;

    tracestruct_names = get_processed_traces(A, win_unit, num_units);
    A.trace_structs = tracestruct_names;

    fprintf('Done processing Traces!\n');

    %% Get df/f for full movie:

    df_min = 20;

    get_df_traces(A, df_min);

    save(fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source)), '-struct', 'A', '-append')
end
