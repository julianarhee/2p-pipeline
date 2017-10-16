%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Select TIFF dirs for current analysis

% Run INIT_HEADER.m

init_header

%%

% Generate analysis-specific info struct:

datetime = strsplit(analysis_id, ' ');
rundate = datetime{1};

I = struct();
I.mc_id = mc_id;
I.roi_method = roi_method; %mcparams.method;
I.roi_id = roi_id;
I.corrected = curr_mcparams.corrected;
I.mc_method = curr_mcparams.method;
I.use_bidi_corrected = use_bidi_corrected; 
if isempty(slices)
    I.slices = A.slices;
else
    I.slices = slices;
end
I.functional = tiff_source;
I.signal_channel = signal_channel;

itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});

path_to_fn = fullfile(acquisition_base_dir, 'analysis_info.txt');
path_to_analysisinfo_json = fullfile(acquisition_base_dir, 'analysis_info.json');

analysisinfo_fn = dir(path_to_fn);
if isempty(analysisinfo_fn)
    % Create new:
    path_to_fn = fullfile(acquisition_base_dir, 'analysis_info.txt');
    itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});
    writetable(itable, path_to_fn, 'Delimiter', '\t', 'WriteRowNames', true);
    new_info = true;
else
    existsI = readtable(path_to_fn, 'Delimiter', '\t', 'ReadRowNames', true);
    %prevruns = existsI.Properties.RowNames;
    %updatedI = [existsI; itable];
    %writetable(updatedI, path_to_fn, 'Delimiter', '\t', 'WriteRowNames', true);
    new_info = false;
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

if ~isfield(A, 'acquisition_base_dir')
    A.acquisition_base_dir = acquisition_base_dir;
end
if isfield(A, 'data_dir') && ~ismember(data_dir, A.data_dir)
    A.data_dir{end+1} = data_dir;
else
    A.data_dir = {fullfile(A.acquisition_base_dir, A.functional, 'DATA')};
end

funcdir_idx = find(arrayfun(@(c) any(strfind(A.data_dir{c}, I.functional)), 1:length(A.data_dir))); 


if do_preprocessing

      % -------------------------------------------------------------------------
      %% 1.  Set MC params
      % -------------------------------------------------------------------------
 
    if isfield(A, 'use_bidi_corrected')
         A.use_bidi_corrected = unique([A.use_bidi_corrected I.use_bidi_corrected]);
    else
         A.use_bidi_corrected = I.use_bidi_corrected;                              % Extra correction for bidi-scanning for extracting ROIs/traces (set mcparams.bidi_corrected=true)
    end 
    if isfield(A, 'signal_channel') 
        A.signal_channel = unique([A.signal_channel I.signal_channel]);                                      % If multi-channel, Ch index for extracting activity traces
    else
        A.signal_channel = I.signal_channel;
    end
        
    if isfield(A, 'corrected')
        A.corrected = unique([A.corrected curr_mcparams.corrected]);
    else
        A.corrected = I.corrected; 
    end
    if isfield(A, 'mc_id')
        A.mc_id = unique([A.mc_id I.mc_id]);
    else
        A.mc_id = I.mc_id;
    end

    save(path_to_reference, '-struct', 'A', '-append');

    % TODO (?):  Run meta-data parsing after
    % flyback-correction (py), including SI-meta correction if
    % flyback-correction changes the TIFF volumes.

    % -------------------------------------------------------------------------
    %% 2.  Do Motion-Correction (and/or) Get Slice t-series:
    % -------------------------------------------------------------------------

    if I.corrected && new_mc_id
        do_motion_correction = true; 
    elseif I.corrected && ~new_mc_struct
        found_nchannels = dir(fullfile(A.data_dir{funcdir_idx}, curr_mcparams.corrected_dir, '*Channel*'));
        found_nchannels = {found_nchannels(:).name}';
        if isdir(fullfile(A.data_dir{funcdir_idx}, curr_mcparams.corrected_dir, found_nchannels{1}))
            found_nslices = dir(fullfile(A.data_dir{funcdir_idx}, curr_mcparams.corrected_dir, found_nchannels{1}, '*.tif'));
            found_nslices = {found_nslices(:).name}';
            if found_nchannels==A.nchannels && found_nslices==length(A.nslices)
                fprintf('Found corrected number of deinterleaved TIFFs in Corrected dir.\n');
                user_says_mc = input('Do Motion-Correction agai
            if strcmp(user_says_mc, 'Y')
                do_motion_correction = true;
            elseif strcmp(user_says_mc, 'n')
                do_motion_correction = false;
            end
            
        else
            fprintf('Found these TIFFs in Corrected dir:\n');
            found_nchannels
            user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's')
            if strcmp(user_says_mc, 'Y')
                do_motion_correction = true;
            elseif strcmp(user_says_mc, 'n')
                do_motion_correction = false;
            end
        end
    else
        % Just parse raw tiffs:
        do_motion_correction = false;
    end

    if do_motion_correction
        [A, curr_mcparams] = motion_correct_data(A, curr_mcparams);
        fprintf('Completed motion-correction!\n');
    else
        fprintf('Not doing motion-correction...\n');
        if ~I.corrected
            fprintf('Parsing RAW tiffs into ./DATA/Parsed.\n');
            [A, curr_mcparams] = create_deinterleaved_tiffs(A, curr_mcparams, I.functional);
       end
    end
    mcparams.(mc_id) = curr_mcparams;
    save(A.mcparams_path, '-struct', 'mcparams', '-append');
    
    fprintf('Finished Motion-Correction step.\n');

    % -------------------------------------------------------------------------
    %% 3.  Clean-up and organize corrected TIFFs into file hierarchy:
    % -------------------------------------------------------------------------
    % mcparams.split_channels = true;
        
    % TODO:  recreate too-big-TIFF error to make a try-catch statement that
    % re-interleaves by default, and otherwise splits the channels if too large
   
    if ~exist(fullfile(curr_mcparams.tiff_dir, 'Raw'), 'dir')
        mkdir(fullfile(curr_mcparams.tiff_dir, 'Raw'));
    end
    uncorrected_tiff_fns = dir(fullfile(curr_mcparams.tiff_dir, '*.tif'));
    uncorrected_tiff_fns = {uncorrected_tiff_fns(:).name}'
    for movidx=1:length(uncorrected_tiff_fns)
        %[datadir, fname, ext] = fileparts(obj.Movies{movidx});
        movefile(fullfile(curr_mcparams.tiff_dir, uncorrected_tiff_fns{movidx}), fullfile(curr_mcparams.tiff_dir, 'Raw', uncorrected_tiff_fns{movidx}));
    end
    fprintf('Moved %i files into ./DATA/Raw before reinterleaving.\n', length(uncorrected_tiff_fns));


    
    % PYTHON equivalent faster: 
    if I.corrected
        deinterleaved_source = curr_mcparams.corrected_dir;
    else
        deinterleaved_source = curr_mcparams.parsed_dir;
    end
    deinterleaved_tiff_dir = fullfile(A.data_dir{funcdir_idx}, deinterleaved_source);
    reinterleave_tiffs(A, deinterleaved_tiff_dir, A.data_dir{funcdir_idx}, curr_mcparams.split_channels);

    % Sort parsed slices by Channel-File:
    path_to_cleanup = fullfile(mcparams.tiff_dir, mcparams.corrected_dir);
    post_mc_cleanup(path_to_cleanup, A);


    % -------------------------------------------------------------------------
    %% 4.  Do additional bidi correction (optional)
    % -------------------------------------------------------------------------
    % mcparams.bidi_corrected = true;

    if I.use_bidi_corrected
        [A, curr_mcparams] = do_bidi_correction(A, I, curr_mcparams);
    end
    
    % Sort bidi-corrected:
    if I.use_bidi_corrected %isfield(mcparams, 'bidi_corrected_dir')
        path_to_cleanup = fullfile(curr_mcparams.tiff_dir, curr_mcparams.bidi_corrected_dir);
        post_mc_cleanup(path_to_cleanup, A);     
    end

    % -------------------------------------------------------------------------
    %% 5.  Create averaged slices from desired source:
    % -------------------------------------------------------------------------
    %A.use_bidi_corrected = false;
    if I.corrected && I.use_bidi_corrected
        new_average_source_dir = curr_mcparams.bidi_corrected_dir;
    elseif I.corrected && ~I.use_bidi_corrected
        new_average_source_dir = curr_mcparams.corrected_dir;
    elseif ~I.corrected
        new_average_source_dir = curr_mcparams.parsed_dir;
    end
    if isfield(A, 'source_to_average')
        A.source_to_average{end+1} = new_average_source_dir;
    else
         A.source_to_average = {new_average_source_dir};
    end
    I.average_source = new_average_source_dir;
                 
    source_tiff_basepath = fullfile(curr_mcparams.tiff_dir, I.average_source);
    dest_tiff_basepath = fullfile(curr_mcparams.tiff_dir, sprintf('Averaged_Slices_%s', I.average_source));
    curr_mcparams.averaged_slices_dir = dest_tiff_basepath;

    mcparams.(mc_id) = curr_mcparams; 
    save(A.mcparams_path, '-struct', 'mcparams', '-append');
 
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
    savejson('', I, path_to_analysisinfo_json);

    fprintf('DONE!\n');

end


%% Update info table again:
itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});
updatedI = update_analysis_table(updatedI, itable, path_to_fn);
