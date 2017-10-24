
% This script checks workspace for user-specified params for analysis.
% Either loads existing analysis-id (from <acquistion_dir>/analysis_record.json), which loads:
%       I :  current 'rolodex' entry that contains params specified in init_header.m
%       A :  full reference struct (<acquisition_dir>/reference_functional.mat) that is first created by python meta-extraction script (process_raw.py).
%            Reference struct 'A' should contain additional fields corresponding to params that are specific to a given rolodex entry.
% ...or...
%   1. Builds all paths from user-specified params in init_header.m
%   2. Adds fields to reference struct 'A' that are common across specific analysis-IDs (if they don't already exist, which is the case if first analysis).
%   3. Checks user-specific preprocessing params (curr_mcparams) against existing (generically stored in <acquisition_dir>/<functional_folder_name>/DATA/mcparams.mat) 
%   4. Creates new mcparmas file, new mcparams entry ('mc_id'), or, if user-specified params match existing entry, asks if matching entry should be reused.
%   5. Checks fields of curr_mcparams to make sure they are consistent with user-provided params in init_header.m

    
path_to_rolodex = fullfile(source, experiment, session, acquisition, 'analysis_record.json');
path_to_rolodex_table = fullfile(source, experiment, session, acquisition, 'analysis_record.txt');
if ~exist(path_to_rolodex, 'file')
    fprintf('No existing analysis record found... Creating new.\n');
    new_rolodex = true;
else
    new_rolodex = false;
end

if ~new_rolodex && load_analysis
    rolodex = loadjson(path_to_rolodex);
    existing_analyses = fieldnames(rolodex);
    fprintf('Found existing analyses:\n');
    if length(existing_analyses)>0
        while (1)
            for a=1:length(existing_analyses)
                fprintf('%i: %s\n', a, existing_analyses{a});
            end
            selected_analysis_idx = input('Enter IDX of analysis_id to use: \n');
            fprintf('Selected the following analysis method:\n');
            display(rolodex.(existing_analyses{selected_analysis_idx}));
            user_confirm = input('Press <A> to accept, otherwise hit <enter> to re-select: \n', 's');
            if strcmp(user_confirm, 'A')
                I = rolodex.(existing_analyses{selected_analysis_idx});
                break;
            else
                fprintf('RETRY.\n')
            end
        end
        new_analysis = false;
        new_mc_id = false;
        new_rolodex_entry = false; 

        % Build acq path and get reference struct:
        % ----------------------------------------
        acquisition_base_dir = fullfile(source, experiment, session, acquisition)
        path_to_reference = fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source))
        path_to_reference_json = fullfile(acquisition_base_dir, sprintf('reference_%s.json', tiff_source))

        A = load(path_to_reference);
        
        % Load MCPARAMS:
        mcparams = load(A.mcparams_path);
        curr_mcparams = mcparams.(I.mc_id);

    else
        fprintf('Rolodex has no fields. Creating new.\n');
        new_rolodex = true;
        new_rolodex_entry = true;
        
    end
    

else
    
    new_rolodex_entry = true;

    % Go through the steps to check MCparams & check if new analysis entry should be created:

    %% 1. Specify paths:
    if ~useGUI 
        curr_tiff_dir = fullfile(source, experiment, session, acquisition, tiff_source);
    else
        default_root = '/nas/volume1/2photon/projects';             % DIR containing all experimental data
        tiff_dirs = uipickfiles('FilterSpec', default_root);        % Returns cell-array of full paths to selected folders containing TIFFs to be processed
        
        % For now, just do 1 dir, but later can iterate over each TIFF dir
        % selected:
        curr_tiff_dir = tiff_dirs{1};

    end
    
    % Iterate through selected tiff-folders to build paths:
    % TODO:  do similar selection step for PYTHON (Step1 preprocessing)

    acquisition_base_dir = fullfile(source, experiment, session, acquisition);
     
    [tiff_parent, tiff_source, ~] = fileparts(curr_tiff_dir);
    [acq_parent, acquisition, ~] = fileparts(tiff_parent);
    [sess_parent, session, ~] = fileparts(acq_parent);
    [source, experiment, ~] = fileparts(sess_parent);

    acquisition_base_dir = fullfile(source, experiment, session, acquisition)
    path_to_reference = fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source))
    path_to_reference_json = fullfile(acquisition_base_dir, sprintf('reference_%s.json', tiff_source))

    %% 2. Load reference struct (initially generated when extracting raw TIFF metadata in python/preprocessing/process_raw.py):
    % ----------------------------------------
 
    A = load(path_to_reference);

    data_dir = fullfile(acquisition_base_dir, tiff_source, 'DATA');

    % Update REFERENCE struct with current analysis-info:
    if ~isfield(A, 'acquisition_base_dir')
        A.acquisition_base_dir = acquisition_base_dir;
    end
    if ~isfield(A, 'mcparams_path')
        A.mcparams_path = fullfile(data_dir, 'mcparams.mat');
    end
    if ~isfield(A, 'roi_dir')
        A.roi_dir = fullfile(A.acquisition_base_dir, 'ROIs'); %, A.roi_id, 'roiparams.mat');
    end
    if ~isfield(A, 'trace_dir')
        A.trace_dir = fullfile(A.acquisition_base_dir, 'Traces'); %, A.trace_id);
    end


    % -------------------------------------------------------------------------
    %% 3.  Set MC params
    % -------------------------------------------------------------------------
    % NOTE:  These params can be reset repeatedly, followed by calling run_pipeline.m
    % TODO:  Autogenerate a new identifier for each parameter change. Append all mods 
    % to reference struct and update.

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
        'corrected', correct_motion,...
        'method', method,...
        'flyback_corrected', flyback_corrected,...
        'ref_channel', reference_channel,...
        'ref_file', reference_file,...
        'algorithm', algorithm,...
        'split_channels', split_channels,...
        'source_dir', data_dir,...
        'nchannels', A.nchannels);

    % Correct mcaram fields based on user-input:
    if correct_bidi_scan
        curr_mcparams.bidi_corrected = true;
    else
        curr_mcparams.bidi_corrected = false;
    end
    if correct_motion
        curr_mcparams.corrected = true
        curr_mcparams.dest_dir = 'Corrected';
    else
        curr_mcparams.dest_dir = 'Raw';
        curr_mcparams.corrected = false;
        curr_mcparams.method = 'None';
        curr_mcparams.algorithm = 'None';
        curr_mcparams.ref_channel = 0;
        curr_mcparams.ref_file = 0;
    end

    %% 4. Check if new mcparams or not:
    fields_to_check = {'corrected', 'method', 'source_dir', 'flyback_corrected', 'ref_channel', 'ref_file', 'algorithm', 'bidi_corrected', 'split_channels', 'crossref'};

    new_mc_id = false;
    if exist(fullfile(data_dir, 'mcparams.mat'))
        mcparams = load(fullfile(data_dir, 'mcparams.mat'))
        new_mc_file = false;

        curr_fieldnames = fields_to_check; %fieldnames(curr_mcparams);
        prev_mcparams_ids = fieldnames(mcparams);
        if length(prev_mcparams_ids)>0
            num_mc_ids = length(prev_mcparams_ids);
            new_mc = [];
            for mc_idx = 1:length(prev_mcparams_ids)
                curr_mcparams_id = prev_mcparams_ids{mc_idx};
                for mf=1:length(curr_fieldnames)
                    tmp_prev.(curr_fieldnames{mf}) = mcparams.(curr_mcparams_id).(curr_fieldnames{mf});
                    tmp_curr.(curr_fieldnames{mf}) = curr_mcparams.(curr_fieldnames{mf});
                end
                if isequal(tmp_prev, tmp_curr)
                    new_mc = [new_mc mc_idx];
               end
            end
            if any(new_mc)
                new_mc_id = false;
            else
                new_mc_id = true;
            end
        else
            mcparams = struct();
            new_mc_file = true;
            new_mc_id = true;
            num_mc_ids = 0;
        end
    else
        mcparams = struct();
        new_mc_file = true;
        new_mc_id = true;
        num_mc_ids = 0;
    end

    % Create NEW mcparams entry, or load previous one, if reusing:
    if new_mc_id
        fprintf('Creating NEW mc struct.\n');
        mc_id = sprintf('mcparams%02d', num_mc_ids+1);
        mcparams.(mc_id) = curr_mcparams;
        save(A.mcparams_path, '-struct', 'mcparams');
    else
        while (1)
            fprintf('Found previous mcstruct with specified params:\n')
            for midx=1:length(prev_mcparams_ids)
                fprintf('%i, %s\n', midx, prev_mcparams_ids{midx});
            end
            user_selected_mc = input('Enter IDX of mcparams struct to view:\n');
            mcparams.(prev_mcparams_ids{user_selected_mc})
            confirm_selection = input('Use these params? Press Y/n.\n', 's');
            if strcmp(confirm_selection, 'Y')
                mc_id = prev_mcparams_ids{user_selected_mc}
                curr_mcparams = mcparams.(mc_id);
                
                break;
            end
        end
    end

    %% 5. Fix base mc dir to allow for multiple mcparams 'CorrectedXX' dirs

    if process_raw
        if correct_motion && ~any(strfind(curr_mcparams.dest_dir, mc_id))
            curr_mcparams.dest_dir = sprintf('%s_%s', curr_mcparams.dest_dir, mc_id)
        end
    else
        if ~exist('processed_source', 'var')
            fprintf('Specified non-raw source for processing, but did not specify folder name.\n');
            processed_source = input('Type FOLDER name of source (ex: Corrrected_mcparams03): \n', 's');
        end
        while (1)
            if ~exist(fullfile(curr_mcparams.source_dir, processed_source)) || length(dir(fullfile(curr_mcparams.source_dir, processed_source, '*.tif')))==0
                fprintf('Processed-source dir is empty or does not exist: %s.\n', processed_source);
                processed_source = input('Try again: ', 's');
            else
                break
            end
        end
        curr_mcparams.dest_dir = sprintf('%s', processed_source);
    end

    if correct_motion && ~(strcmp(average_source, curr_mcparams.dest_dir))
        average_source = curr_mcparams.dest_dir;
    end
    if correct_bidi_scan && ~(strcmp(average_source, sprintf('%s_Bidi', curr_mcparams.dest_dir)))
        average_source = sprintf('%s_Bidi', curr_mcparams.dest_dir);
    end
%end

    % Resave mcparams:
    if exist(A.mcparams_path, 'file')
        fprintf('Loading and writing to MCPARAMS path.\n');
        mcparams = load(A.mcparams_path);
    else
        fprintf('Creating new MCPARAMS.\n');
        mcparams = struct();
    end
    mcparams.(mc_id) = curr_mcparams;
    save(A.mcparams_path, '-struct', 'mcparams');

end
