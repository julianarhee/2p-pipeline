% Put in .gitignore at top of repo. Edit to run specific acquisitions.

% Add *_header.m to /2p-pipeline/.gitignore file after cloning repo.
clear all; clc;

% ============================================================================
% USER-SPECIFIED INFO:
% ============================================================================

% Specify what to run:
useGUI = false;                 % Must specify acquisition-path info if false
load_analysis = false;          % true, to reload existing analysis (e.g., to do ROI/Trace extraction)

% Set info manually:
source = '/nas/volume1/2photon/projects';
experiment = 'gratings_phaseMod'; 
session = '20171024_CE062';
acquisition = 'FOV1'; 
tiff_source = 'functional'; 

% ----------------------------------------------------------------------------
% Set the following if NOT loading a previous analysis:
% ----------------------------------------------------------------------------
process_raw = false;             % false, if source for processing is anything but RAW 
processed_source = '';          % Folder name contaning CORRECTED tiffs, if processing on non-raw source


% Set ROI params: 
roi_id = ''; %'blobDoG01'; 
roi_method = ''; %'pyblob2D'; %'manual2D_circles'
;
% Specify what to run it on:
slices = [];                    % List of slice indices (e.g., [5, 10, 15, 20, 25, 30, 35, 40])
signal_channel = 1;             % If multi-channel, Ch index for extracting activity traces
flyback_corrected = false;      % true, if python process_raw.py --correct-flyback
split_channels = false;

% Set Motion-Correction params:
correct_motion = true; 
correct_bidi_scan = true;       % true, if want to fix artifacts from idirectional scanning
reference_channel = 1;
reference_file = 9;             % File00X to use as reference for motion-correction %6; %3; %6; %3
method = 'Acquisition2P';       % [opts: 'Acquisition2P', 'NoRMCorre'] 
algorithm = @withinFile_withinFrame_lucasKanade; % [opts: @lucasKanade_plus_nonrigid, @withinFile_withinFrame_lucasKanade || 'rigid', 'nonrigid'

 
% These vars are checked/corrected once mcparam set is identified, not as critical:
average_source = 'Raw';         % FINAL output type ['Corrected', 'Parsed', 'Corrected_Bidi']

analysis_id = 'analysis02';
% ----------------------------------------------------------------------------
% ============================================================================
    
