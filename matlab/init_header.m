% Put in .gitignore at top of repo. Edit to run specific acquisitions.

% Add *_header.m to /2p-pipeline/.gitignore file after cloning repo.

useGUI = false; 
get_rois_and_traces = true %false;
do_preprocessing = false %true;

slices = [];
signal_channel = 1;                                 % If multi-channel, Ch index for extracting activity traces


if ~useGUI 
    % Set info manually:
    source = '/nas/volume1/2photon/projects';
    experiment = 'gratings_phaseMod';
    session = '20170927_CE059';
    acquisition = 'FOV1_zoom3x'; %'FOV1_zoom3x';
    tiff_source = 'functional'; %'functional_subset';
    acquisition_base_dir = fullfile(source, experiment, session, acquisition);
    curr_tiff_dir = fullfile(acquisition_base_dir, tiff_source);
else
    default_root = '/nas/volume1/2photon/projects';             % DIR containing all experimental data
    tiff_dirs = uipickfiles('FilterSpec', default_root);        % Returns cell-array of full paths to selected folders containing TIFFs to be processed
    
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
path_to_reference_json = fullfile(acquisition_base_dir, sprintf('reference_%s.json', tiff_source))

A = load(path_to_reference);

data_dir = fullfile(acquisition_base_dir, tiff_source, 'DATA');

if ~isfield(A, 'mcparams_path')
    A.mcparams_path = fullfile(data_dir, 'mcparams.mat');    % Standard path to mcparams struct (don't change)
end

analysis_id = datestr(now())
datetime = strsplit(analysis_id, ' ');
rundate = datetime{1};


% -------------------------------------------------------------------------
%% 1.  Set MC params
% -------------------------------------------------------------------------
% NOTE:  These params can be reset repeatedly, followed by calling run_pipeline.m
% TODO:  Autogenerate a new identifier for each parameter change. Append all mods 
% to reference struct and update.

use_bidi_corrected = false;                              % Extra correction for bidi-scanning for extracting ROIs/traces (set mcparams.bidi_corrected=true)

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

curr_mcparams = set_mc_params(...
    'corrected', true,...
    'method', 'Acquisition2P',...
    'flyback_corrected', false,...
    'ref_channel', 1,...
    'ref_file', 6,...
    'algorithm', @withinFile_withinFrame_lucasKanade,...
    'split_channels', false,...
    'bidi_corrected', true,... %,...
    'tiff_dir', data_dir,...
    'nchannels', A.nchannels);              

%% 2. ROI parmas:

roi_method = 'pyblob2D';
roi_id = 'blobs_DoG';

%% 3.  Check if new mcparams or not:
new_mc_id = false;
if exist(fullfile(data_dir, 'mcparams.mat'))
    mcparams = load(fullfile(data_dir, 'mcparams.mat'));

    curr_fieldnames = fieldnames(curr_mcparams);
    prev_mcparams_ids = fieldnames(mcparams);
    if length(prev_mcparams_ids)>0
        num_mc_ids = length(prev_mcparams_ids);
        for mc_idx = 1:length(prev_mcparams_ids)
            curr_mcparams_id = prev_mcparams_ids{mc_idx};
            prev_fieldnames = fieldnames(mcparams.(curr_mcparams_id)); 
            for mf=1:length(curr_fieldnames)
                if ~ismember(curr_fieldnames{mf}, prev_fieldnames)
                    new_mc_id = true;
                else
                    if isstr(curr_mcparams.(curr_fieldnames{mf}))
                        if ~strcmp(curr_mcparams.(curr_fieldnames{mf}), mcparams.(curr_mcparams_id).(curr_fieldnames{mf}))
                            new_mc_id = true;
                        end
                    elseif isa(curr_mcparams.(curr_fieldnames{mf}), 'function_handle')
                        if ~strcmp(char(curr_mcparams.(curr_fieldnames{mf})), char(mcparams.(curr_mcparams_id).(curr_fieldnames{mf})))
                            new_mc_id = true;
                        end
                    else
                        if curr_mcparams.(curr_fieldnames{mf}) ~= mcparams.(curr_mcparams_id).(curr_fieldnames{mf})
                            new_mc_id = true;
                        end
                    end
                end
            end
        end
    else
        mcparams = struct();
        save(A.mcparams_path, '-struct', 'mcparams');
        new_mc_id = true;
        num_mc_ids = 0;    
    end
else
    mcparams = struct();
    save(A.mcparams_path, '-struct', 'mcparams');
    new_mc_id = true;
    num_mc_ids = 0;
end

if new_mc_id
    mc_id = sprintf('mcparams%02d', num_mc_ids+1);
    mcparams.(mc_id) = curr_mcparams;
    save(A.mcparams_path, '-struct', 'mcparams', '-append');
else
    while (1) 
        fprintf('Found previous mcstruct with specified params:\n')
        for midx=1:length(prev_mcparams_ids)
            fprintf('%i, %s', midx, prev_mcparams_ids{midx});
        end
        user_selected_mc = input('Enter IDX of mcparams struct to view:\n');
        mcparams.(prev_mcprams_ids{user_selected_mc})
        confirm_selection = input('Use these params? Press Y/n.\n');
        if strcmp(confirm_selection, 'Y')
            mc_id = prev_mcparams_ids{user_selected_mc});
            break;
        end
    end
end


 
