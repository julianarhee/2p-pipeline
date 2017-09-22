%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Create Acquisition Struct:

A.source = '/nas/volume1/2photon/projects/gratings_phaseMod';
A.session = '20170901_CE054_zoom3x';
A.base_dir = fullfile(A.source, A.session);
A.acquisition = 'functional_test';

A.preprocess = true;
A.correct_bidi = true;

A.slices
A.signal_channel

%% Specify MC param struct path:
if A.preprocess
    A.tiff_source = fullfile(A.base_dir, A.acquisition, 'DATA');
else
    A.tiff_source = fullfile(A.base_dir, A.acquisition);
end
A.mcparams_path = fullfile(A.tiff_source, 'mcparams.mat');

%% Preprocess data:

% do_preprocessing();
test_preprocessing_steps;


%% Specify ROI param struct path:
A.roi_id = 'blobs_DoG';

A.roiparams_path = fullfile(A.base_dir, 'ROIs', A.roi_id, 'roiparams.mat');

%% GET ROIS.


%% Specify Traces param struct path:

A.trace_id = 'blobs_DoG';
A.trace_dir = fullfile(A.base_dir, 'Traces', A.trace_id);
if ~exist(A.trace_dir, 'dir')
    mkdir(A.trace_dir)
end

%% Get traces
extract_traces(A);

%% GET metadata for SI tiffs:

si = get_scan_info(A, mcparams)
save(fullfile(A.tiff_source, 'simeta.mat'), 'si', '-struct');
A.simeta_path = fullfile(A.tiff_source, 'simeta.mat');

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



