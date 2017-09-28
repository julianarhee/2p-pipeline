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
    experiment = 'gratings_phaseMod';
    session = '20170927_CE059';
    acquisition = 'FOV1_zoom3x';
    tiff_source = 'functional';
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
A = load(fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source)));
A

if do_preprocessing
    A.acquisition_base_dir = acquisition_base_dir;
    A.data_dir = fullfile(A.acquisition_base_dir, A.functional, 'DATA');

    A
    %% Set MC-related params:
    % ----------------------------------------

    A.use_bidi_corrected = true;                                               % Use standard-corrected or extra-bidi-corrected files for processing traces/ROIs (must set mcparams.bidi_corrected=true)
    A.signal_channel = 1;                                                      % If multi-channel, Ch index for extracting activity traces

    A.mcparams_path = fullfile(A.data_dir, 'mcparams.mat');                    % Standard path to mcparams struct

    % Names = [
    %     'corrected          '       % corrected or raw (T/F)
    %     'method             '       % Source for doing correction. Can be custom. ['Acqusition2P', 'NoRMCorre']
    %     'flyback_corrected  '       % True if did correct_flyback.py 
    %     'ref_channel        '       % Ch to use as reference for correction
    %     'ref_file           '       % File index (of numerically-ordered TIFFs) to use as reference
    %     'algorithm          '       % Depends on 'method': Acq_2P [@lucasKanade_plus_nonrigid, @withinFile_withinFrame_lucasKanade], NoRMCorre ['rigid', 'nonrigid']
    %     'split_channels     '       % *MC methods should parse corrected-tiffs by Channel-File-Slice (Acq2P does this already). Last step interleaves parsed tiffs, but sometimes they are too big for Matlab
    %     'bidi_corrected     '       % *For faster scanning, SI option for bidirectional-scanning is True -- sometimes need extra scan-phase correction for this
    %     ];

    mcparams = set_mc_params(...
        'corrected', 'true',...
        'method', 'Acquisition2P',...
        'flyback_corrected', true,...
        'ref_channel', 1,...
        'ref_file', 6,...
        'algorithm', @lucasKanade_plus_nonrigid,...
        'split_channels', false,...
        'bidi_corrected', true,...
        'tiff_dir', A.data_dir,...
        'nchannels', A.nchannels);                                                             % Edit this file to populate MC-param fields
        

    save(fullfile(A.data_dir, 'mcparams.mat'), 'mcparams');

    % TODO (?):  Run meta-data parsing after
    % flyback-correction (py), including SI-meta correction if
    % flyback-correction changes the TIFF volumes.

    %% Preprocess data:

    preprocess_data(mcparams);
    load(A.mcparams_path) % Load mcparams again to get updated struct info
    generate_slice_images(mcparams);


    fprintf('Finished preprocessing data.\n');
    fprintf('MC params saved:\n');
    mcparams

    save(fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source)), '-struct', 'A', '-append')
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
