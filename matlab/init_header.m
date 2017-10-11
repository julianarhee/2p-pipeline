% Put in .gitignore at top of repo. Edit to run specific acquisitions.

% Add *_header.m to /2p-pipeline/.gitignore file after cloning repo.

useGUI = false;
get_rois_and_traces = false %true %false;
do_preprocessing = true %false %true;

if ~useGUI
    % Set info manually:
    source = '/nas/volume1/2photon/projects';
    experiment = 'gratings_phaseMod';
    session = '20171009_CE059';
    acquisition = 'FOV1_zoom3x'; %'FOV1_zoom3x';
    tiff_source = 'functional'; %'functional_subset';
    acquisition_base_dir = fullfile(source, experiment, session, acquisition);
    curr_tiff_dir = fullfile(acquisition_base_dir, tiff_source);
end


signal_channel = 1;                                      % If multi-channel, Ch index for extracting activity traces

data_dir = fullfile(acquisition_base_dir, tiff_source, 'DATA');

% -------------------------------------------------------------------------
%% 1.  Set MC params
% -------------------------------------------------------------------------
% NOTE:  These params can be reset repeatedly, followed by calling run_pipeline.m
% TODO:  Autogenerate a new identifier for each parameter change. Append all mods 
% to reference struct and update.

use_bidi_corrected = false;                              % Extra correction for bidi-scanning for extracting ROIs/traces (set mcparams.bidi_corrected=true)

% set_mc_params():

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
    'flyback_corrected', false,...
    'ref_channel', 1,...
    'ref_file', 6,...
    'algorithm', @withinFile_withinFrame_lucasKanade,...
    'split_channels', false,...
    'bidi_corrected', true); %,...
%    'tiff_dir', data_dir,...
%    'nchannels', A.nchannels);              

