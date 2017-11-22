%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Create Acquisition Struct:

source = '/nas/volume1/2photon/projects';
experiment = 'gratings_phaseMod';
session = '20170901_CE054';
acquisition = 'FOV1_zoom3x';
datadir = 'functional_sub';
base_dir = fullfile(source, experiment, session, acquisition);

analysis_name = strjoin({session, acquisition, }, '_')

filename = fullfile(base_dir, analysis_name);
proplist = 'H5P_DEFAULT';
% 'H5F_ACC_TRUNC' - to overwrite
% 'H5F_ACC_EXCL' - don't overwrite existing
fid = H5F.create(filename,'H5F_ACC_TRUNC', proplist, proplist);
gid = H5G.create(fid, 'source', proplist, proplist, proplist);
H5G.close(gid);

h5writeatt(filename, '/', 'created_on', datestr(now));
h5writeatt(filename, '/', 'source', source);
h5writeatt(filename, '/', 'experiment', experiment);
h5writeatt(filename, '/', 'session', session);
h5writeatt(filename, '/', 'acquisition', acquisition);
h5writeatt(filename, '/', 'data', datadir);


attr = 'acquisition_info';
attr_details.Name = 'Paths';
attr_details.AttachedTo = '/source'
attr_details.AttachType = 'dataset';
hdf5write(filename, attr_details, attr, 'WriteMode', 'append');

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



