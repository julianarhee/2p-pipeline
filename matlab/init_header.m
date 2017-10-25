% Put in .gitignore at top of repo. Edit to run specific acquisitions.

% Add *_header.m to /2p-pipeline/.gitignore file after cloning repo.
clear all; clc;

% ============================================================================
% USER-SPECIFIED INFO:
% ============================================================================

% Specify what to run:
useGUI = false;                                     % Must specify acquisition-path info if false
load_analysis = false; %true;                              % True if want to reload existing analysis to complete ROI/Trace extraction

% Set info manually:
source = '/nas/volume1/2photon/projects';
experiment = 'gratings_phaseMod'; %'scenes'; %'gratings_phaseMod';
session = '20171023_CE062';
acquisition = 'FOV1'; %'FOV1_zoom3x';
tiff_source = 'functional'; %'functional_subset';

% ----------------------------------------------------------------------------
% Set the following if NOT loading a previous analysis:
% ----------------------------------------------------------------------------

analysis_id = '';

% Set ROI params: 
roi_id = 'blobDoG01'; %manual2D_poly_aligned'; %'pyblob2D';
roi_method = 'pyblob2D'; %'manual2D_poly_aligned'; %'blobs_DoG';

% Specify what to run it on:
slices = []; %[5, 10, 15, 20, 25, 30, 35, 40];
signal_channel = 1;                                 % If multi-channel, Ch index for extracting activity traces
flyback_corrected = false; %false;
split_channels = false;

% Set Motion-Correction params:
correct_motion = true; %false;
correct_bidi_scan = true; %true; %true; %false;
reference_channel = 1;
reference_file = 6; %3; %6; %3
method = 'Acquisition2P'; 
algorithm = @withinFile_withinFrame_lucasKanade; %@lucasKanade_plus_nonrigid; %@withinFile_withinFrame_lucasKanade

% These vars are checked/corrected once mcparam set is identified, not as critical:
average_source = 'Raw';                             % FINAL output type ['Corrected', 'Parsed', 'Corrected_Bidi']
process_raw = false;                                % True if not re-using previous corrected dirs
processed_source = '';                              % Folder name contaning CORRECTED tiffs, if want to do processing on non-raw source

% ----------------------------------------------------------------------------
% ============================================================================
    
