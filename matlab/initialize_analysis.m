% function initialize_analysis()
%% Clear all and make sure paths set
%clc; clear all;

% add repo paths
%add_repo_paths
%fprintf('Added repo paths.\n');

%if new_rolodex_entry

I = struct();
I.mc_id = mc_id;
I.corrected = curr_mcparams.corrected;
I.mc_method = curr_mcparams.method;
if isempty(roi_id)
    I.roi_id = 'None';
else
    I.roi_id = roi_id;
end
if isempty(roi_method);
    I.roi_method = 'None';
else
    I.roi_method = roi_method;
end
if isempty(slices)
    I.slices = A.slices;
else
    I.slices = slices;
end
I.functional = tiff_source;
I.signal_channel = signal_channel;
I.average_source = average_source;

%itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});

if load_analysis
    I.analysis_id = analysis_id;
    entries = fieldnames(tmpI);
    if ~isequal(tmpI, I)
        fprintf('Found mismatch in current rolodex entry specifications (init_header) and loaded entry selected...\n');
        for e=1:length(entries)
            if isstr(tmpI.(entries{e})) && ~strcmp(tmpI.(entries{e}), I.(entries{e}))
                fprintf('%s:\nPrev: %s, Curr: %s\n', entries{e}, tmpI.(entries{e}), I.(entries{e}));
                choice = input('Select <0> to keep old, <1> to overwrite with curr: ');
                if choice==1
                    tmpI.(entries{e}) = I.(entries{e});
                end
            end
        end
    end
    I = tmpI;
    rolodex.(existing_analyses{selected_analysis_idx}) = I;
    savejson('', rolodex, path_to_rolodex);

    % Update rolodex table:
    itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});
    updated_records = update_analysis_table(itable, path_to_rolodex_table);

    % Also store as json:
    existing_analyses = updated_records.Properties.RowNames;
    existing_records = table2struct(updated_records);

    varnames = fieldnames(I);
    n_analysis_ids = length(existing_analyses)+1;
    if exist('updated_records', 'var'), clear updated_records, end
    for var=1:length(varnames)
        updated_records(selected_analysis_idx).(varnames{var}) = I.(varnames{var});
    end
    for id=1:length(existing_analyses)
        for var=1:length(varnames)
            updated_records(id).(varnames{var}) = existing_records(id).(varnames{var});
        end
    end
    for id=1:length(updated_records)
        json_records.(updated_records(id).analysis_id) = updated_records(id);
    end
    savejson('', json_records, path_to_rolodex);

% 
%     fprintf('Loaded previous analysis entry, with ID: %s\n', I.analysis_id);
%     display(I);
%     display(curr_mcparams);
% 


else
    if ~new_rolodex && ~exist('rolodex', 'var') % i.e., not using previous analysis
        
        existing_records = readtable(path_to_rolodex_table, 'Delimiter', '\t', 'ReadRowNames', true);
        existing_analysis_names = existing_records.Properties.RowNames;
        existing_records = table2struct(existing_records);
        existing_analysis_idxs = [1:length(existing_records)]; %fieldnames(existing_records);
        
        % Check if current analysis exists:
        tmpI = struct();
        curr_fields = fieldnames(I);
        for field=1:length(fieldnames(I))
            curr_subfield = curr_fields{field};
            if any(size(I.(curr_subfield))>1) && ~ischar(I.(curr_subfield))
                tmpI.(curr_subfield) = mat2str(I.(curr_subfield));
            else
                tmpI.(curr_subfield) = I.(curr_subfield);
            end
        end
        for aid = 1:length(existing_analysis_idxs)
            tmp_record = rmfield(existing_records(aid), 'analysis_id'); 
            if isequal(tmp_record, tmpI)
                new_analysis = false;
                curr_analysis_idx = aid;
            else
                new_analysis = true;
                curr_analysis_idx = length(existing_analysis_idxs) + 1;
            end
        end
    else
        existing_analysis_idxs = {};
        curr_analysis_idx = 1;
        new_analysis = true;
    end

    % Check if user-provided analysis ID already exists:
    if isempty(analysis_id)
        analysis_id = sprintf('analysis%02d', curr_analysis_idx);
    else
        overwrite = false;
        while (1)
            if ismember(analysis_id, existing_analysis_names) && ~overwrite
                fprintf('User provided analysis name that already exists.\n');
                eidx = find(arrayfun(@(i) strcmp(analysis_id, existing_analysis_names{i}), 1:length(existing_analysis_names)));
                display(existing_records(eidx))
                user_says_create = input('Create new analysis ID? Press Y/n: ', 's');
                if strcmp(user_says_create, 'Y')
                    fprintf('Existing names:\n')
                    display(existing_analysis_names);
                    user_analysis_id = input('Enter new IDX, or enter new analysis name (add F to force overwrite):\n', 's')
                    if strfind(user_analysis_id, 'F')
                        confirm_overwrite_id = input('Enforcing ovewrite... Re-enter analysis idx to continue: \n', 's');
                        if all(ismember(confirm_overwrite_id, '0123456789+-.eEdD'))
                            analysis_id = sprintf('analysis%02d', str2num(confirm_overwrite_id));
                            overwrite = true
                            new_analysis = true;
                            break;
                        end
                    elseif all(ismember(user_analysis_id, '0123456789+-.eEdD'))
                        analysis_id = sprintf('analysis%02d', str2num(user_analysis_id));
                    else
                        analysis_id = user_analysis_id;
                    end
                    %end
                    new_analysis = true;
                else
                    user_says_reuse = input(sprintf('Use old analysis ID: %s? Press Y/n: ', analysis_id), 's');
                    if user_says_reuse
                        new_analysis = false;
                        break;
                    end 
                end
            else
                fprintf('Creating NEW analysis with id: %s\n', analysis_id);
                break;
            end
        end
    end
    I.analysis_id = analysis_id;
    itable = struct2table(I, 'AsArray', true, 'RowNames', {analysis_id});

    if new_analysis
        update_analysis_table(itable, path_to_rolodex_table);

        % Also store as json:
        varnames = fieldnames(I);
        n_analysis_ids = length(existing_analysis_idxs)+1;
        if exist('updated_records', 'var'), clear updated_records, end
        for var=1:length(varnames)
            updated_records(n_analysis_ids).(varnames{var}) = I.(varnames{var});
        end
        for id=1:length(existing_analysis_idxs)
            for var=1:length(varnames)
                updated_records(id).(varnames{var}) = existing_records(id).(varnames{var});
            end
        end
        for id=1:length(updated_records)
            json_records.(updated_records(id).analysis_id) = updated_records(id);
        end
        savejson('', json_records, path_to_rolodex);
    end
end

% Note:  This step also adds fields to mcparams struct that are immutable,
% i.e., standard across all analyses.
 
% These fields get updated based on current analysis settings:
if ~isfield(A, 'data_dir')
    A.data_dir = struct();
end
A.data_dir.(I.analysis_id) = data_dir;
if ~isfield(A, 'bidi')
     A.bidi = struct();
end
A.bidi.(I.analysis_id) = curr_mcparams.bidi_corrected; 
if ~isfield(A, 'signal_channel') 
    A.signal_channel = struct();
end
A.signal_channel.(I.analysis_id) = I.signal_channel;        
if ~isfield(A, 'corrected')
    A.corrected = struct();
end
A.corrected.(I.analysis_id) = curr_mcparams.corrected;
if ~isfield(A, 'mc_id')
    A.mc_id = struct();
end
A.mc_id.(I.analysis_id) = I.mc_id;
if ~isfield(A, 'average_source')
    A.average_source = struct();
end
A.average_source.(I.analysis_id) = I.average_source;

% ROI step:
if ~isfield(A, 'roi_id')
    A.roi_id = struct();
end
A.roi_id.(I.analysis_id) = I.roi_id;
if ~isfield(A, 'roi_method')
    A.roi_method = struct();
end
A.roi_method.(I.analysis_id) = I.roi_method;
    
% TRACES step:
if ~isfield(A, 'trace_id')
    A.trace_id = struct();
end
if isempty(I.roi_id)
    A.trace_id.(I.analysis_id) = '';
else
    A.trace_id.(I.analysis_id) = strjoin({I.roi_id, I.mc_id}, '/');
end
if ~isfield(A, 'simeta_path')
    A.simeta_path = struct();
end
A.simeta_path.(I.analysis_id) = fullfile(A.data_dir.(I.analysis_id), 'simeta.mat');


% save updated reference struct:
save(path_to_reference, '-struct', 'A'); %, '-append');
savejson('', A, path_to_reference_json);

%end

fprintf('Updated analysis-info files. Udpated acquisition reference files.\n');
fprintf('Ready for ROI extraction or trace extraction.\n');
display(I)
display(curr_mcparams)


%% PREPROCESSING:  motion-correction.

% 1. Set parameters for motion-correction.
% 2. If correcting, make standard directories and run correction.
% 3. Re-interleave parsed TIFFs and save in tiff base dir (child of
% functional-dir). Sort parsed TIFFs by Channel-File-Slice.
% 4. Do additional correction for bidirectional scanning (and do
% post-mc-cleanup). [optional]
% 5. Create averaged time-series for all slices using selected correction
% method. Save in standard Channel-File-Slice format.

% 
% if do_preprocessing
% 
%     % TODO (?):  Run meta-data parsing after
%     % flyback-correction (py), including SI-meta correction if
%     % flyback-correction changes the TIFF volumes.
% 
%     % -------------------------------------------------------------------------
%     %% 2.  Do Motion-Correction (and/or) Get Slice t-series:
%     % -------------------------------------------------------------------------
%     if I.corrected && new_mc_id && process_raw
%         do_motion_correction = true; 
%     elseif I.corrected && (~new_mc_id || ~process_raw)
%         found_nchannels = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, '*Channel*'));
%         found_nchannels = {found_nchannels(:).name}';
%         if length(found_nchannels)>0 && isdir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, found_nchannels{1}))
%             found_nslices = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, found_nchannels{1}, '*.tif'));
%             found_nslices = {found_nslices(:).name}';
%             if length(found_nchannels)==A.nchannels && length(found_nslices)==length(A.nslices)
%                 fprintf('Found corrected number of deinterleaved TIFFs in Corrected dir.\n');
%                 user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's')
%             end
%             if strcmp(user_says_mc, 'Y')
%                 do_motion_correction = true;
%                 user_says_delete = input('Deleting old MC folder tree. Press Y to confirm:\n', 's');
%                 if strcmp(user_says_delete, 'Y')
%                     rmdir fullfile(mcparams.source_dir, curr_mcparams.dest_dir) s
%                 end
%             elseif strcmp(user_says_mc, 'n')
%                 do_motion_correction = false;
%             end 
%         else
%             found_tiffs = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, '*.tif'));
%             found_tiffs = {found_tiffs(:).name}';
%             fprintf('Found these TIFFs in Corrected dir - %s:\n', curr_mcparams.dest_dir);
%             found_tiffs
%             user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's')
%             if strcmp(user_says_mc, 'Y')
%                 do_motion_correction = true;
%             elseif strcmp(user_says_mc, 'n')
%                 do_motion_correction = false;
%             end
%         end
%     else
%         % Just parse raw tiffs:
%         do_motion_correction = false;
%     end
%     
%     % CHECK anyway:
%     if do_motion_correction
%         % This is redundant to above, but double-check temporarily: 
%         found_nfiles = dir(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir, '*.tif'));
%         found_nfiles = {found_nfiles(:).name}';
%         if length(found_nfiles)==A.ntiffs 
%             fprintf('Found correct number of interleaved TIFFs in Corrected dir: %s\n', curr_mcparams.dest_dir);
%             user_says_mc = input('Do Motion-Correction again? Press Y/n.\n', 's');
%         end
%         if strcmp(user_says_mc, 'Y')
%             do_motion_correction = true;
%             user_says_delete = input('Deleting old MC folder tree. Press Y to confirm:\n', 's');
%             if strcmp(user_says_delete, 'Y')
%                 rmdir fullfile(mcparams.source_dir, curr_mcparams.dest_dir) s
%             end
%         elseif strcmp(user_says_mc, 'n')
%             do_motion_correction = false;
%         end 
%     end 
% 
% 
%     if do_motion_correction
%         % Do motion-correction on "raw" tiffs in ./DATA, save slice tiffs to curr_mcparams.dest_dir
%         % Rename deinterleaved-corrected TIFFs to ./DATA/<curr_mcparams.dest_dir>_slices in next step.
%         % Interleave and save interleaved-corrected TIFFs to ./DATA/<curr_mcparams.dest_dir. in next step.
%         curr_mcparams = motion_correct_data(curr_mcparams);
%         fprintf('Completed motion-correction!\n');
%     else
%         % Just parse "raw" tiffs in ./DATA to ./DATA/Raw_slices, then move original tiffs to ./DATA/Raw
%         fprintf('Not doing motion-correction...\n');
%         curr_mcparams = create_deinterleaved_tiffs(A, curr_mcparams);
%     end
%    
%     fprintf('Finished Motion-Correction step.\n');
% 
%     % -------------------------------------------------------------------------
%     %% 3.  Clean-up and organize corrected TIFFs into file hierarchy:
%     % -------------------------------------------------------------------------
%     % mcparams.split_channels = true;
%         
%     % TODO:  recreate too-big-TIFF error to make a try-catch statement that
%     % re-interleaves by default, and otherwise splits the channels if too large
%   
%     % Reinterleave slices tiffs into full Files (only if ran motion-correction):
%     % Don't need to reinterleave and sort TIFFs if just parsing raw, since that is done in create_deinterleaved_tiffs()
%  
%     % PYTHON equivalent faster?: 
%     deinterleaved_tiff_dir = fullfile(curr_mcparams.source_dir, sprintf('%s_slices', curr_mcparams.dest_dir));
%     if I.corrected && process_raw 
%         if ~isdir(deinterleaved_tiff_dir)
%             movefile(fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir), deinterleaved_tiff_dir);
%         end
%         reinterleaved_tiff_dir = fullfile(curr_mcparams.source_dir, curr_mcparams.dest_dir);
%         reinterleave_tiffs(A, deinterleaved_tiff_dir, reinterleaved_tiff_dir, curr_mcparams.split_channels);
%     end
%     
%     % Sort parsed slices by Channel-File:
%     post_mc_cleanup(deinterleaved_tiff_dir, A);
% 
%     fprintf('Finished Motion-Correction step.\n');
% 
%     % -------------------------------------------------------------------------
%     %% 4.  Do additional bidi correction (optional)
%     % -------------------------------------------------------------------------
%     % mcparams.bidi_corrected = true;
% 
%     if curr_mcparams.bidi_corrected
%         if curr_mcparams.corrected
%             if process_raw
%                 bidi_source = sprintf('%s_%s', 'Corrected', I.mc_id);
%             else
%                 bidi_source = curr_mcparams.dest_dir;
%             end
%         else
%             bidi_source = 'Raw';
%         end
%         curr_mcparams = do_bidi_correction(bidi_source, curr_mcparams, A);
%     end
% 
%     % NOTE: bidi function updates mcparams.dest_dir 
%     deinterleaved_tiff_dir = fullfile(curr_mcparams.source_dir, sprintf('%s_slices', curr_mcparams.dest_dir));
% 
%     % Sort parsed slices by Channel-File:
%     post_mc_cleanup(deinterleaved_tiff_dir, A);
% 
%     fprintf('Finished BIDI correction step.\n')
% 
%     % -------------------------------------------------------------------------
%     %% 5.  Create averaged slices from desired source:
%     % -------------------------------------------------------------------------
%                 
%     source_tiff_basepath = fullfile(curr_mcparams.source_dir, sprintf('%s_slices', I.average_source));
%     dest_tiff_basepath = fullfile(curr_mcparams.source_dir, sprintf('Averaged_Slices_%s', I.average_source));
% 
%     create_averaged_slices(source_tiff_basepath, dest_tiff_basepath, A);
%     fprintf('Finished creating AVERAGED slices from dir %s\n', I.average_source);
% 
%     % Update mcparams struct and reference struct:
%     curr_mcparams.averaged_slices_dir = dest_tiff_basepath;
%   
%     % Also save MCPARAMS as json:
%     [ddir, fname, ext] = fileparts(A.mcparams_path);
%     mcparams_json = fullfile(ddir, strcat(fname, '.json'));
%     %savejson('', mcparams, mcparams_json);    
%     % TODO:  this does not save properly of n-fields (mcparam ids) > 2...
%  
%     save(path_to_reference, '-struct', 'A', '-append')
%     
%     % Also save json:
%     savejson('', A, path_to_reference_json);
% 
%     fprintf('Finished preprocessing data.\n');
% 
%     % -------------------------------------------------------------------------
%     %% 6.  Create updated simeta struct, if doesn't exist
%     % -------------------------------------------------------------------------
%     if ~exist(fullfile(curr_mcparams.source_dir, sprintf('%s.mat', A.base_filename)), 'file')
%         raw_simeta = load(A.raw_simeta_path);
%         metaDataSI = {};
%         for fi=1:A.ntiffs
%             currfile = sprintf('File%03d', fi);
%             tmpSI = raw_simeta.(currfile);
%             if curr_mcparams.flyback_corrected
%                 curr_simeta = adjust_si_metadata(tmpSI, mov_size);
%             else
%                 curr_simeta = tmpSI;
%             end
%             metaDataSI{fi} = curr_simeta;
%         end 
%         eval([A.base_filename '=struct();'])
%         eval([(A.base_filename) '.metaDataSI = metaDataSI'])
%         mfilename = sprintf('%s', A.base_filename);
%         save(fullfile(curr_mcparams.source_dir, mfilename), mfilename);
%         fprintf('Created updated SI metadata for flyback-corrected TIFFs to:\n')
%         fprintf('%s\n', fullfile(curr_mcparams.source_dir, mfilename));
%     end
%             
% end
% 
% mcparams.(mc_id) = curr_mcparams;
% save(A.mcparams_path, '-struct', 'mcparams');
% 

% %% Specify ROI param struct path:
% if get_rois_and_traces
%     %% GET ROIS.
% 
% 
%     %% Specify Traces param struct path:
%     curr_trace_dir = fullfile(A.trace_dir, A.trace_id.(I.analysis_id));
%     if ~exist(curr_trace_dir)
%         mkdir(curr_trace_dir)
%     end
% 
%     %% Get traces
%     fprintf('Extracting RAW traces.\n')
%     extract_traces(I, curr_mcparams, A);
%     fprintf('Extracted raw traces.\n')
% 
%     %% GET metadata for SI tiffs:
% 
%     si = get_scan_info(I, A)
%     save(A.simeta_path.(I.analysis_id), '-struct', 'si');
%  
% 
%     %% Process traces
%     fprintf('Doing rolling mean subtraction.\n')
% 
%     % For retino-movie:
%     % targetFreq = meta.file(1).mw.targetFreq;
%     % winUnit = (1/targetFreq);
%     % crop = meta.file(1).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
%     % nWinUnits = 3;
% 
%     % For PSTH:
%     win_unit = 3;   % size of one window (sec)
%     num_units = 3;  % number of win_units that make up sliding window 
% 
%     tracestruct_names = get_processed_traces(I, A, win_unit, num_units);
%     %A.trace_structs = tracestruct_names;
% 
%     fprintf('Done processing Traces!\n');
% 
%     %% Get df/f for full movie:
%     fprintf('Getting df/f for file.\n');
% 
%     df_min = 50;
% 
%     get_df_traces(I, A, df_min);
% 
% 
%     save(path_to_reference, '-struct', 'A', '-append')
%     
%     % Also save json:
%     savejson('', A, path_to_reference_json);
% 
%     fprintf('DONE!\n');
% 
% end
% 

