%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Select TIFF dirs for current analysis

% Run INIT_HEADER.m

init_header

if useGUI
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
path_to_reference_json = fullfile(acquisition_base_dir, sprintf('reference_%s.json', tiff_source))

A = load(path_to_reference);

% TODO:  Load (or generate) "analysis-trail" file to store all processing steps and iterations


%%

datetime = strsplit(analysis_id, ' ');
rundate = datetime{1};

I = struct();
I.roi_method = roi_method; mcparams.method;
I.roi_id = roi_id;
I.corrected = mcparams.corrected;
I.mc_method = mcparams.method;
I.use_bidi_corrected = use_bidi_corrected; 
if isempty(slices)
    I.slices = A.slices;
else
    I.slices = slices;
end
itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});

path_to_fn = fullfile(acquisition_base_dir, 'analysis_info.txt');
path_to_analysisinfo_json = fullfile(acquisition_base_dir, 'analysis_info.json');

analysisinfo_fn = dir(path_to_fn);
if isempty(analysisinfo_fn)
    % Create new:
    path_to_fn = fullfile(acquisition_base_dir, 'analysis_info.txt');
    itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});
    writetable(itable, path_to_fn, 'Delimiter', '\t', 'WriteRowNames', true);
else
    existsI = readtable(path_to_fn, 'Delimiter', '\t', 'ReadRowNames', true);
    %prevruns = existsI.Properties.RowNames;
    %updatedI = [existsI; itable];
    %writetable(updatedI, path_to_fn, 'Delimiter', '\t', 'WriteRowNames', true);
end

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
     mcparams.tiff_dir = A.data_dir;
     mcparams.nchannels = A.nchannels;              

      % -------------------------------------------------------------------------
      %% 1.  Set MC params
      % -------------------------------------------------------------------------
 
    if isfield(A, 'use_bidi_corrected')
         A.use_bidi_corrected = unique([A.use_bidi_corrected I.use_bidi_corrected]);
    else
         A.use_bidi_corrected = I.use_bidi_corrected;                              % Extra correction for bidi-scanning for extracting ROIs/traces (set mcparams.bidi_corrected=true)
    end 
    A.signal_channel = signal_channel;                                      % If multi-channel, Ch index for extracting activity traces
        
    if isfield(A, 'corrected')
        A.corrected = unique([A.corrected mcparams.corrected]);
    else
        A.corrected = I.corrected; 
    end
    A.mcparams_path = fullfile(A.data_dir, 'mcparams.mat');    % Standard path to mcparams struct (don't change)
    save(A.mcparams_path, 'mcparams');
    save(path_to_reference, '-struct', 'A', '-append');

    % TODO (?):  Run meta-data parsing after
    % flyback-correction (py), including SI-meta correction if
    % flyback-correction changes the TIFF volumes.

    % -------------------------------------------------------------------------
    %% 2.  Do Motion-Correction (and/or) Get Slice t-series:
    % -------------------------------------------------------------------------

    if I.corrected
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
   
    if ~exist(fullfile(mcparams.tiff_dir, 'Raw'), 'dir')
        mkdir(fullfile(mcparams.tiff_dir, 'Raw'));
    end
    uncorrected_tiff_fns = dir(fullfile(mcparams.tiff_dir, '*.tif'));
    uncorrected_tiff_fns = {uncorrected_tiff_fns(:).name}'
    for movidx=1:length(uncorrected_tiff_fns)
        %[datadir, fname, ext] = fileparts(obj.Movies{movidx});
        movefile(fullfile(mcparams.tiff_dir, uncorrected_tiff_fns{movidx}), fullfile(mcparams.tiff_dir, 'Raw', uncorrected_tiff_fns{movidx}));
    end
    fprintf('Moved %i files into ./DATA/Raw before reinterleaving.\n', length(uncorrected_tiff_fns));


    
    % PYTHON equivalent faster: 
    if I.corrected
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

    if I.use_bidi_corrected
        [A, mcparams] = do_bidi_correction(A, I, mcparams);
    end
    
    % Sort bidi-corrected:
    if I.use_bidi_corrected %isfield(mcparams, 'bidi_corrected_dir')
        path_to_cleanup = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
        post_mc_cleanup(path_to_cleanup, A);     
    end

    % -------------------------------------------------------------------------
    %% 5.  Create averaged slices from desired source:
    % -------------------------------------------------------------------------
    %A.use_bidi_corrected = false;
    if I.corrected && I.use_bidi_corrected
        new_average_source_dir = mcparams.bidi_corrected_dir;
    elseif I.corrected && ~I.use_bidi_corrected
        new_average_source_dir = mcparams.corrected_dir;
    elseif ~I.corrected
        new_average_source_dir = mcparams.parsed_dir;
    end
    if isfield(A, 'source_to_average')
        A.source_to_average{end+1} = new_average_source_dir;
    else
         A.source_to_average = {new_average_source_dir};
    end
    I.average_source = new_average_source_dir;
                 
%     if I.corrected
%         if I.use_bidi_corrected                                % TODO: may want to have intermediate step to evaluate first MC step...
%             A.source_to_average = mcparams.bidi_corrected_dir;
%         else
%             A.source_to_average = mcparams.corrected_dir;
%         end
%     else
%         A.source_to_average = mcparams.parsed_dir;               % if no correction is done in preprocessing step above, still parse tiffs by slice to get t-series
%     end
 
    source_tiff_basepath = fullfile(mcparams.tiff_dir, I.average_source);
    dest_tiff_basepath = fullfile(mcparams.tiff_dir, sprintf('Averaged_Slices_%s', I.average_source));
    mcparams.averaged_slices_dir = dest_tiff_basepath;
    save(A.mcparams_path, 'mcparams', '-append');
 
    create_averaged_slices(source_tiff_basepath, dest_tiff_basepath, I, A);
    save(path_to_reference, '-struct', 'A', '-append')

    % Also save json:
    savejson('', A, path_to_reference_json);
    savejson('', I, path_to_analysisinfo_json);

    fprintf('Finished preprocessing data.\n');

end

%% Update itable:

I.roi_method = roi_method;
I.roi_id = roi_id;

itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});
updatedI = update_analysis_table(existsI, itable, path_to_fn);


%% Specify ROI param struct path:
if get_rois_and_traces
    if isfield(A, 'roi_method')
        A.roi_method{end+1} = I.roi_method;
    else
        A.roi_method = {I.roi_method};
    end
    A.roi_method = unique(A.roi_method);
    if isfield(A, 'roi_id')
        A.roi_id{end+1} = I.roi_id;
    else
        A.roi_id = {I.roi_id};
    end
    A.roi_id = unique(A.roi_id);
    
    %A.roi_method = 'pyblob2D';
    %A.roi_id = 'blobs_DoG';

    A.roi_dir = fullfile(A.acquisition_base_dir, 'ROIs'); %, A.roi_id, 'roiparams.mat');

    %% GET ROIS.


    %% Specify Traces param struct path:
    if isfield(A, 'trace_id')
        A.trace_id{end+1} = I.roi_id;
    else
        A.trace_id = {I.roi_id}; %'blobs_DoG';
    end
    A.trace_id = unique(A.trace_id);
    A.trace_dir = fullfile(A.acquisition_base_dir, 'Traces'); %, A.trace_id);
    curr_trace_dir = fullfile(A.trace_dir, I.roi_id);
%     if ~exist(A.trace_dir, 'dir')
%         mkdir(A.trace_dir)
%     end
    if ~exist(curr_trace_dir)
        mkdir(curr_trace_dir)
    end

    %% Get traces
    extract_traces(I, A);
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

    tracestruct_names = get_processed_traces(I, A, win_unit, num_units);
    %A.trace_structs = tracestruct_names;

    fprintf('Done processing Traces!\n');

    %% Get df/f for full movie:

    df_min = 50;

    get_df_traces(I, A, df_min);


    save(path_to_reference, '-struct', 'A', '-append')
    
    % Also save json:
    savejson('', A, path_to_reference_json);
    fprintf('DONE!\n');

end


%% Update info table again:
itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});
updatedI = update_analysis_table(updatedI, itable, path_to_fn);
