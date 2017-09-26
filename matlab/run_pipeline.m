%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Create Acquisition Struct:

A.source = '/nas/volume1/2photon/projects/gratings_phaseMod';
A.session = '20170901_CE054';
A.acquisition = 'FOV1_zoom3x';
A.functional = 'functional_sub';

A.acquisition_base_dir = fullfile(A.source, A.session, A.acquisition);
A.data_dir = fullfile(A.acquisition_base_dir, A.functional, 'DATA');

A.use_bidi_corrected = true;
A.signal_channel = 1;


% TODO:  This (and other) info should just be read from SI TIFFs
% themselves, instead of user-input. Run meta-data parsing after
% flyback-correction (py), including SI-meta correction if
% flyback-correction changes the TIFF volumes.
A.slices = [1:13];
A.nchannels = 2;

%% Specify MC param struct path:

A.mcparams_path = fullfile(A.data_dir, 'mcparams.mat');

%% Preprocess data:

% do_preprocessing();
test_preprocessing_steps;


%% Save n files:
if A.use_bidi_corrected
    base_slice_dir = fullfile(A.data_dir, mcparams.bidi_corrected_dir, sprintf('Channel%02d', A.signal_channel));
elseif A.corrected && ~A.correct_bidi
    base_slice_dir = fullfile(A.data_dir, mcparams.corrected_dir, sprintf('Channel%02d', A.signal_channel));
else
    base_slice_dir = fullfile(A.data_dir, mcparams.parsed_dir, sprintf('Channel%02d', A.signal_channel));
end
file_dirs = dir(fullfile(base_slice_dir, 'File*'));
file_dirs = {file_dirs(:).name}';
A.ntiffs = length(file_dirs);

%% Specify ROI param struct path:

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



