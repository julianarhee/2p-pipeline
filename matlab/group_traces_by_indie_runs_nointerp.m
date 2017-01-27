
clear all;
clc

%% Set paths for current acquisition:

%
% 
% source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
% mw_path = strcat(source_dir, 'mw_data/');
% pymat_fn = '20161222_JR030W_gratings_10reps_run1.mat';
% 
% 
% source_slice_no = 8;
% source_file_no = 2;
% source_struct_fn = strcat(source_dir, sprintf('slice%i_traces_CC.mat', source_slice_no));
% source_struct = load(source_struct_fn);
% colors = source_struct.D.(['slice' num2str(source_slice_no)]).(['file' num2str(source_file_no)]).colors;
% %
% 
% slice_idx = 8;
% run_idx = 2;
% source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
% tiff_dir = strcat(source_dir, 'Corrected/');
% slice_no = sprintf('slice%i', slice_idx);
% 
% % -------------------------------------------------------------------------
% % Set here only for source files to be used for getting masks:
% curr_tiff = sprintf('fov1_gratings_10reps_run1_Slice%02d_Channel01_File%03d.tif', slice_idx, run_idx);
% curr_tiff_path = strcat(tiff_dir, curr_tiff);
% 
% struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')
% load(struct_fn)

%% Set paths for acquis.
%
% 
source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/';
mw_path = strcat(source_dir, 'mw_data/');
pymat_fn = '20160118_AG33_gratings_fov1_run10.mat';

S = load(strcat(source_dir, 'mw_data/', pymat_fn));

% -------------------------------------------------------------------------
% Set here only for source files to be used for getting masks:
source_slice_no = 6;
source_file_no = 10;
source_struct_fn = strcat(source_dir, sprintf('slice%i_traces_AVG.mat', source_slice_no));
source_struct = load(source_struct_fn);
colors = source_struct.D.(['slice' num2str(source_slice_no)]).(['file' num2str(source_file_no)]).colors;
%


% -------------------------------------------------------------------------
% Set here for all other runs relevant for above source dir:
slice_idx = 6; % default source
run_idx = 10;
tiff_dir = strcat(source_dir, 'ch1_slice6/');
slice_no = sprintf('slice%i', slice_idx);

% reload D (source struct, but also contains processed files in
% fieldnames):
%struct_fn = strcat(source_dir, slice_no, '_traces_AVG_singleROI.mat')
struct_fn = strcat(source_dir, slice_no, '_traces_AVG.mat')
load(struct_fn)

% -------------------------------------------------------------------------
% TIFF naming + relevant path for all other tifs to be l =oaded::
curr_tiff = sprintf('fov1_gratings_%05d.tif #2.tif #%i.tif', run_idx, slice_idx);
curr_tiff_path = strcat(tiff_dir, curr_tiff);


%% Load time series:
nam = strcat(tiff_dir, curr_tiff);

sframe=1;						% user input: first frame to read (optional, default 1)

Y = bigread2(nam,sframe);
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels


%% Get pymat info:
% 
% Get MW and ARD info from python .mat file:

interpolate_y = 0;
separate_runs = 1;

if separate_runs == 0

    pymat_fn = '20160118_AG33_gratings_fov1_run10.mat';
    S = load(strcat(source_dir, 'mw_data/', pymat_fn));

    run_idx = 2;
    ntrash = 1;
    start_tif = 7;
    tif_no = start_tif - ntrash;
    ndiscard = 5;
    mc_file_no = 1;
    mw_fidx = 6 + run_idx - 1; % tif_no - ntrash; %mc_file_no+ntrash;

else
    
    run_fns = dir(strcat(source_dir, 'mw_data/*.mat'));
    curr_run_suffix = sprintf('_run%i.mat', run_idx);

    runs_to_use = [1 2 3 4 5 6 7 8 10];
    trial_struct = struct();
    
    for rtu=1:length(runs_to_use)

    % Get n-stim color-code: -- get 'mw_codes' from match_triggers.py:
    nstimuli = 35; %length(unique(curr_mw_codes)-1;
    % nstimuli = 12;


    % Sort into trials by MW code:
    % For each .mwk & corresponding .tif stacks, do:
    % 1.  Split time series into trials -- MW times used as splitter.
    % 2.  For each motion-corrected tif file, there should be a corresponding
    % MW file (or part of a file) that contains the MW stimulus codes for each
    % time-defined trial (in mw_times). Use these codes as fieldnames for the
    % trial-struct associated with the current experiment run. Sort occurrences
    % of each trial by stimulus type (i.e., code number). 
    % 3.  Append relative onset times for each occurrence of a given stimulus, 
    % using the appropriate MW index and corresponding motion-corrected tif 
    % index (File00x.tif).

    acquisitions = fieldnames(D.(slice_no));
    %trial_struct = struct();
    
    stim_types = cell(1,length(S.gratings));
    for sidx=1:length(S.gratings)
        sname = sprintf('code%i', sidx);
        stim_types{sidx} = sname;
    end
    %[stim_types,~] = sort_nat(fieldnames(trial_struct));
    
    % ndiscard = 5;
    % ntrash = 1;

    %for curr_run=1:length(acquisitions) %nruns

    %file_no = acquisitions{curr_run}; %sprintf('file%i', curr_run);
    
    file_no = sprintf('file%i', runs_to_use(rtu));
    
    %traces = D.(slice_no).(file_no).rolltraces; %D.(slice_no).(file_no).traces;
    %traces = D.(slice_no).(file_no).deltaF; %D.(slice_no).(file_no).traces;
    traces = D.(slice_no).(file_no).raw_traces;
    %active_cells = D.(['slice', num2str(source_slice_no)]).(['file' num2str(source_file_no)]).active_cells;


    %trial_struct = struct();

    %for tif=1:ntifs
        %mw_fidx = curr_run + abs_mw_fidx - 1;
        curr_run_suffix = sprintf('_run%i.mat', runs_to_use(rtu));
        for pyfn=1:length(run_fns)
            if strfind(run_fns(pyfn).name, curr_run_suffix)
                pymat_fn = run_fns(pyfn).name
            end
        end
        %end
        mw_fidx = 1;

        % Need to specify how many secs after onset of PREVIOUS stimulus to use
        % as baseline for given trial.  For 1sec ON, 2sec OFF -- 1 sec. For
        % 1sec ON 9sec OFF -- 8 sec (to just have 1 sec pre):
        nsecs_pre = 7; 

        %end
        S = load(strcat(source_dir, 'mw_data/', pymat_fn));

        mw_times = S.mw_times_by_file;
        offsets = double(S.offsets_by_file);
        mw_codes = S.mw_codes_by_file;
        mw_file_durs = double(S.mw_file_durs);
        if isfield(S, 'ard_file_durs')
            ard_times = S.ard_file_trigger_times;
            ard_filedurs = double(S.ard_file_durs);
        end
        
        
        %mw_file_no = sprintf('mw%ifile%i', mw_fidx, curr_run);
        mw_file_no = sprintf('mw%i%s', mw_fidx, file_no);
        if strcmp(class(mw_times), 'int64')
            curr_mw_times = double(mw_times(mw_fidx, :));       % GET MW rel times from match_triggers.py: 'rel_times'
            curr_mw_codes = mw_codes(mw_fidx, :);               % Get MW codes corresponding to time points:
            fprintf('File %s, Last code found: %i\n', mw_file_no, curr_mw_codes(end-1))

            curr_ard_times = double(ard_times(mw_fidx, :));
        else
            curr_mw_times = double(mw_times{mw_fidx}); % convert to sec
            curr_mw_codes = mw_codes{mw_fidx};
            curr_ard_times = double(ard_times{mw_fidx});
        end
        mw_rel_times = ((curr_mw_times - curr_mw_times(1)) + offsets(mw_fidx)); % into seconds
        mw_sec = mw_rel_times/1000000;

        ard_rel_times = (curr_ard_times - curr_ard_times(1));
        ard_sec = ard_rel_times/1000000;

        % Get correct time dur to convert frames to time (s):
        curr_mw_filedur = mw_file_durs(mw_fidx)/1000000; % convert to sec

        nframes = size(Y, 3);

        % See if ard-file exists and get trigger times from ard file:
        if isfield(S, 'ard_fn')
            ard_filedurs = double(S.ard_file_durs);
            y_sec = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes);
        else
    %         y_sec = [0:nframes-1].*(ard_sec(end)/nframes); %[0:nframes-1].*(curr_mw_filedur/nframes);
            y_sec = interp1([0:length(ard_sec)-1], ard_sec, linspace(0, length(ard_sec)-1, nframes));
        end


        for trial=1:2:length(mw_sec)-1
            code_no = sprintf('code%i', curr_mw_codes(trial));
            if trial==1
                continue;
            end

           % This way of parsing loses last offset of stimulus ON (since py
           % scripts parses by stimulus ID on a trial):

               if isfield(trial_struct, code_no)
                   if isfield(trial_struct.(code_no), mw_file_no)
                       if trial==length(mw_sec)-1
                           trial_struct.(code_no).(mw_file_no){end+1} = mw_sec(trial-1:end);
                       else
                           trial_struct.(code_no).(mw_file_no){end+1} = mw_sec(trial-1:trial+2);
                       end
                   else
                       if trial==length(mw_sec)-1
                            trial_struct.(code_no).(mw_file_no){1} = mw_sec(trial-1:end);
                       else
                           trial_struct.(code_no).(mw_file_no){1} = mw_sec(trial-1:trial+2);
                       end
                   end
               else
                   trial_struct.(code_no) = struct();
                   if trial==length(mw_sec)-1
                       trial_struct.(code_no).(mw_file_no){1} = mw_sec(trial-1:end);
                   else
                       trial_struct.(code_no).(mw_file_no){1} = mw_sec(trial-1:trial+2);
                   end
               end


        end

    %end


    % Use trial_struct & selected ROI mask to plot all reps of each stimulus:


    %t[stim_types,~] = sort_nat(fieldnames(trial_struct));
    for curr_stim=1:length(stim_types)
        
        if ~isfield(trial_struct, stim_types{curr_stim})
            fprintf('File %s: %s not found!\n', mw_file_no, stim_types{curr_stim});
            continue;
        end

        mw_fns = fieldnames(trial_struct.(stim_types{curr_stim}));
        ign_ids = strfind(mw_fns, 'tif');
        mw_fns = mw_fns(find((cellfun('isempty', ign_ids))));

        for mwfn=1:length(mw_fns)

            curr_trials = trial_struct.(stim_types{curr_stim}).(mw_fns{mwfn});
            matching_tiff_fn = strcat('tif_', mw_fns{mwfn});

            for trial=1:length(curr_trials)
                curr_tstamps = trial_struct.(stim_types{curr_stim}).(mw_fns{mwfn}){trial};
                
                if round(curr_tstamps(3)-curr_tstamps(2)) ~= 1
                    fprintf('Displaced tstamps in MW file: %s, %s, trial %i', mw_fns{mwfn}, stim_types{curr_stim}, trial);
                    break
                else
                    t_pre = curr_tstamps(1)+nsecs_pre; % this is actually two seconds before stim onset of next trial
                    t_onset = curr_tstamps(2); 
                    if length(curr_tstamps) < 4 % Last trial of file, missing end tstamp (i.e., no start of next trial)
                        t_post = curr_tstamps(end)+9;
                    else
                        t_post = curr_tstamps(end); 
                    end
                end
                
                if interpolate_y == 1
                    y2 = y_sec(1):(y_sec(2)-y_sec(1))/20:y_sec(end);                % interpolate resampled ard-trigger values to find closer matches to mw-tstamps
                else
                    y2 = y_sec;
                end
                
                y_start_idx = find(abs(y2-t_pre)==min(abs(y2-t_pre)));          % min difference should only be on the order of a few msec
                y_end_idx = find(abs(y2-t_post)==min(abs(y2-t_post)));
                y_onset_idx = find(abs(y2-t_onset)==min(abs(y2-t_onset)));
    %             y_start_idx = find(abs(y_sec-t_pre)==min(abs(y_sec-t_pre)));
    %             y_end_idx = find(abs(y_sec-t_post)==min(abs(y_sec-t_post)));
    %             y_onset_idx = find(abs(y_sec-t_onset)==min(abs(y_sec-t_onset)));

                if trial==1
                    trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){1} = [y_start_idx y_onset_idx y_end_idx];
                else
                    trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){end+1} = [y_start_idx y_onset_idx y_end_idx];
                end
            end

        end
    end



    end
    
end



%% PLOT EACH:



plot_all = 0;

AVG = struct();
for curr_stim=1:length(stim_types)
    
    if ~isfield(trial_struct, stim_types{curr_stim})
        continue
    end
    mw_fns = fieldnames(trial_struct.(stim_types{curr_stim}));
    ign_ids = strfind(mw_fns, 'tif');
    mw_fns = mw_fns(find((cellfun('isempty', ign_ids))));
    AVG.(stim_types{curr_stim}) = struct();
    
    trial_idx = 1;
    for mwfn=1:length(mw_fns) % IF separate runs, mwfn is a trial
        curr_trials = trial_struct.(stim_types{curr_stim}).(mw_fns{mwfn});
        matching_tiff_fn = strcat('tif_', mw_fns{mwfn});
        
        tmp = strsplit(mw_fns{mwfn}, 'file');
        curr_tiff_file_no = ['file' tmp{2}];
        %traces = D.(slice_no).(curr_tiff_file_no).rolltraces; %D.(slice_no).(file_no).traces;
        %active_cells = D.(slice_no).(curr_tiff_file_no).active_cells;
        %traces = D.(slice_no).(curr_tiff_file_no).deltaF; %D.(slice_no).(file_no).traces;
        traces = D.(slice_no).(curr_tiff_file_no).raw_traces;
        
        %trace_mat = zeros(length(active_cells), size(traces,2));
        trace_mat = zeros(size(traces,1), size(y2,2));
        for acell=1:size(traces,1) %length(active_cells)
            %y_sec2 = interp1(y_sec, traces(active_cells(acell),:), y2); % y2 is interpolated for 20x finer sampling
            if interpolate_y==1
                y_trace = interp1(y_sec, traces(acell,:), y2); % y2 is interpolated for 20x finer sampling
            else
                y_trace = traces(acell,:);
            end
            trace_mat(acell,:) = y_trace;
            %trace_mat(acell,:) = traces(active_cells(acell),:);
        end


        for curr_trial=1:length(curr_trials)

            y_start = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(1);
            y_end = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(end);
            y_onset = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(2);
            %centered_y = y_sec(y_start:y_end) - y_sec(y_onset);
            centered_y = y2(y_start:y_end) - y2(y_onset);
            
%             % Tweak if needed:
%             closest_pre = find(abs(centered_y+1)==min(abs(centered_y+1))); % Get corresponding frame-time for "1 sec pre"              
%             y_start = y_start + closest_pre - 1;
            
            % if interpolate, 1000 ... %300 %16
            if length(centered_y) < 50 %1000 %300 %16
                fprintf('STIM: %s, MW file: %s, trial: %i\n', stim_types{curr_stim}, mw_fns{mwfn}, curr_trial);
%                 closest_end = y_sec(y_end)+2; % add 2 sec if a cut-off trial (last trial of file)
%                 closest_end = find(abs(y_sec - closest_end) == min(abs(y_sec - closest_end)));
                closest_end = y2(y_end)+9; % add 2 sec if a cut-off trial (last trial of file)
%                 closest_end = find(abs(y2 - closest_end) == min(abs(y2 - closest_end)));
                y_end = closest_end;
                continue;
            end
            while centered_y(end) > 10 %3
                y_end = y_end - 1;
                centered_y = y2(y_start:y_end) - y2(y_onset);
            end
            
% %             centered_y = y_sec(y_start:y_end) - y_sec(y_onset);            % Crop mw-indexed times to only include 1 sec before (rather than 2 sec)
            centered_y = y2(y_start:y_end) - y2(y_onset);            % Crop mw-indexed times to only include 1 sec before (rather than 2 sec)

            
            % Update any adjustments to first-pass search for matching frame
            % indices;
            trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(1) = y_start;
            trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(end) = y_end;
            trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(2) = y_onset;
            

            
            % Get actual trace values from EACH roi:
            %traces_y = NaN(size(active_cells,1), length(centered_y));
            %deltaFs_y = NaN(size(active_cells,1), length(centered_y));
            traces_y = NaN(size(traces,1), length(centered_y));
            deltaFs_y = NaN(size(traces,1), length(centered_y));
            for active=1:size(traces,1) %length(active_cells)
                if length(centered_y) < 10
                    continue
                end
                traces_y(active,:) = trace_mat(active, y_start:y_end);
                baseline = mean(trace_mat(active, y_start:y_onset-1));
                deltaFs_y(active,:) = ((trace_mat(active, y_start:y_end) - baseline)./baseline) * 100;
            end
            
            
            AVG.(stim_types{curr_stim}).trial_times{trial_idx} = centered_y; %centered_y;
            AVG.(stim_types{curr_stim}).trial_deltaF{trial_idx} = deltaFs_y;
            AVG.(stim_types{curr_stim}).trial_traces{trial_idx} = traces_y;
            trial_idx = trial_idx + 1;
        end
        
    end
   
end


%% PLOT each:

% 
% 
% plot_all = 0;
% 
% AVG = struct();
% for curr_stim=1:length(stim_types)
%     mw_fns = fieldnames(trial_struct.(stim_types{curr_stim}));
%     ign_ids = strfind(mw_fns, 'tif');
%     mw_fns = mw_fns(find((cellfun('isempty', ign_ids))));
%     AVG.(stim_types{curr_stim}) = struct();
%     
% %     if plot_all==1
% %         figure()
% %     end
%     trial_idx = 1;
%     for mwfn=1:length(mw_fns)
%         curr_trials = trial_struct.(stim_types{curr_stim}).(mw_fns{mwfn});
%         matching_tiff_fn = strcat('tif_', mw_fns{mwfn});
%         
%         tmp = strsplit(mw_fns{mwfn}, 'file');
%         curr_tiff_file_no = ['file' tmp{2}];
%         %traces = D.(slice_no).(curr_tiff_file_no).rolltraces; %D.(slice_no).(file_no).traces;
%         %active_cells = D.(slice_no).(curr_tiff_file_no).active_cells;
%         %traces = D.(slice_no).(curr_tiff_file_no).deltaF; %D.(slice_no).(file_no).traces;
%         traces = D.(slice_no).(curr_tiff_file_no).raw_traces;
%         
%         %trace_mat = zeros(length(active_cells), size(traces,2));
%         trace_mat = zeros(size(traces,1), size(y2,2));
%         for acell=1:size(traces,1) %length(active_cells)
%             %y_sec2 = interp1(y_sec, traces(active_cells(acell),:), y2); % y2 is interpolated for 20x finer sampling
%             y_sec2 = interp1(y_sec, traces(acell,:), y2); % y2 is interpolated for 20x finer sampling
%             trace_mat(acell,:) = y_sec2;
%             %trace_mat(acell,:) = traces(active_cells(acell),:);
%         end
% 
% %         trial_colors = zeros(length(curr_trials),3);
% %         for c=1:length(curr_trials)
% %             trial_colors(c,:,:) = rand(1,3);
% %         end
% 
%         for curr_trial=1:length(curr_trials)
% 
%             y_start = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(1);
%             y_end = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(end);
%             y_onset = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(2);
%             %centered_y = y_sec(y_start:y_end) - y_sec(y_onset);
%             centered_y = y2(y_start:y_end) - y2(y_onset);
%             
% %             % Tweak if needed:
% %             closest_pre = find(abs(centered_y+1)==min(abs(centered_y+1))); % Get corresponding frame-time for "1 sec pre"              
% %             y_start = y_start + closest_pre - 1;
%             while centered_y(end) > 10 %3
%                 y_end = y_end - 1;
%                 centered_y = y2(y_start:y_end) - y2(y_onset);
%             end
%             if length(centered_y) < 300 %16
%                 fprintf('STIM: %s, MW file: %s, trial: %i\n', stim_types{curr_stim}, mw_fns{mwfn}, curr_trial);
% %                 closest_end = y_sec(y_end)+2; % add 2 sec if a cut-off trial (last trial of file)
% %                 closest_end = find(abs(y_sec - closest_end) == min(abs(y_sec - closest_end)));
%                 closest_end = y2(y_end)+2; % add 2 sec if a cut-off trial (last trial of file)
% %                 closest_end = find(abs(y2 - closest_end) == min(abs(y2 - closest_end)));
%                 y_end = closest_end;
%                 continue;
%             end
% % %             centered_y = y_sec(y_start:y_end) - y_sec(y_onset);            % Crop mw-indexed times to only include 1 sec before (rather than 2 sec)
%             centered_y = y2(y_start:y_end) - y2(y_onset);            % Crop mw-indexed times to only include 1 sec before (rather than 2 sec)
% 
%             
% %             % Just ignore bad cut-offs for now...
% %             trace_y = trace_mat(y_start:y_end);
% %             if length(trace_y) < 10
% %                 continue
% %             end
% 
%             
%             % Get actual trace values from EACH roi:
%             %traces_y = NaN(size(active_cells,1), length(centered_y));
%             %deltaFs_y = NaN(size(active_cells,1), length(centered_y));
%             traces_y = NaN(size(traces,1), length(centered_y));
%             deltaFs_y = NaN(size(traces,1), length(centered_y));
%             for active=1:size(traces,1) %length(active_cells)
%                 if length(centered_y) < 10
%                     continue
%                 end
%                 traces_y(active,:) = trace_mat(active, y_start:y_end);
%                 baseline = mean(trace_mat(active, y_start:y_onset-1));
%                 deltaFs_y(active,:) = ((trace_mat(active, y_start:y_end) - baseline)./baseline) * 100;
%             end
%             
% %             if plot_all == 1
% %                 subplot(1,2,1)
% %                 plot(centered_y, trace_y)
% %                 y1=get(gca,'ylim');
% %                 title('raw')
% %                 hold on;
% %                 
% % 
% %                 if trial==length(curr_trials)
% %                     centered_mw = curr_trials{trial} - curr_trials{trial}(2);
% %                     sx = [centered_mw(2) centered_mw(3) centered_mw(3) centered_mw(2)];
% %                     sy = [y1(1) y1(1) y1(2) y1(2)];
% %                     patch(sx, sy, colors(curr_stim,:,:), 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
% %                     hold on;
% %                 %if trial==length(curr_trials)
% %                     text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_stim));
% %                     hold on;
% %                 end
% %             end
% 
% %             baseline = mean(trace_mat(y_start:y_onset-1));
% %             deltaF_y = ((trace_mat(y_start:y_end) - baseline)./baseline) * 100;
% %             if plot_all==1
% %                 subplot(1,2,2)
% %                 plot(centered_y, deltaF_y)
% %                 hold on;
% %             end
%             
%             AVG.(stim_types{curr_stim}).trial_times{trial_idx} = centered_y; %centered_y;
%             AVG.(stim_types{curr_stim}).trial_deltaF{trial_idx} = deltaFs_y;
%             AVG.(stim_types{curr_stim}).trial_traces{trial_idx} = traces_y;
%             trial_idx = trial_idx + 1;
%         end
%         
%     end
% %     curr_stim_config = S.gratings(curr_stim,:);
% %     curr_stim_config_name = sprintf('ORI: %0.2f, SF: %0.2f', curr_stim_config(1), curr_stim_config(2));
% %     suptitle(curr_stim_config_name)
% %     
% end

%%  sanity check,

%curr_run = 2;
%file_no = sprintf('file%i', curr_run);
%traces = D.(slice_no).(file_no).rolltraces; %D.(slice_no).(file_no).traces;
%active_cells = D.(['slice', num2str(source_slice_no)]).(['file' num2str(source_file_no)]).active_cells;

active_idx = 1;
roi_idx = 2;
fidx = 1;

%figure()
%t_codes = D_mw_codes(1:2:end);

t_trace= [];
for curr_stim=1:length(stim_types) %length(t_codes)-1
    %t_codes = D_mw_codes(1:2:end);
    %pidx = double(t_codes(curr_stim))
    
    subplot(5,7,curr_stim)
    
    %t_trace = [t_trace AVG.(stim_types{t_codes(curr_stim+1)}).trial_traces{fidx}(active_idx,:)];
    
    %roi_trace = AVG.(stim_types{curr_stim}).trial_traces{fidx}(active_idx,:);
    roi_dF = AVG.(stim_types{curr_stim}).trial_deltaF{fidx}(active_idx,:);
    roi_times = AVG.(stim_types{curr_stim}).trial_times{fidx};
%     if ismember(roi_trace, 'NaN')
%         continue
%     else
        
        %plot(roi_times, roi_trace)
        %hold on;
        plot(roi_times, roi_dF)
        hold on;
        %ylim([-2*(10^5) 3*(10^5)])
        hold on;
        y1=get(gca,'ylim');
        sy = [y1(1) y1(1) y1(2) y1(2)];
        sx = [0 1 1 0];
       %sy = [-0.1 -0.1 0.1 0.1].*100;
        %curr_code = t_codes(curr_stim+1);
        curr_code = stim_types{curr_stim};
       patch(sx, sy, colors(curr_stim,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
       hold on;
       text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_code));

        %title(sprintf('code: %i', t_codes(curr_stim+1)))
        title(sprintf('code: %i', stim_types{curr_stim}))
end


%% Check n frames per trial:

DURS = struct();
for curr_stim=1:length(stim_types)
    
    curr_trials = AVG.(stim_types{curr_stim}).trial_deltaF;
    curr_times = AVG.(stim_types{curr_stim}).trial_times;
    
    curr_frames = [];
    curr_onsets = [];
    for curr_trial = 1:length(curr_trials)
        curr_frames = [curr_frames size(curr_trials{curr_trial},2)];
        curr_onsets = [curr_onsets find(curr_times{curr_trial}==0)];
    end
    
    DURS.nframes{curr_stim} = curr_frames;
    DURS.onsets{curr_stim} = curr_onsets;
    
end

min_frames = [];
max_frames = [];
min_onsets = [];
max_onsets = [];
for stim=1:length(DURS.nframes)
    
    min_frames = [min_frames min(DURS.nframes{stim})];
    max_frames = [max_frames max(DURS.nframes{stim})];
    
    min_onsets = [min_onsets min(DURS.onsets{stim})];
    max_onsets = [max_onsets max(DURS.onsets{stim})];
    
    range_nframes = [unique(min_frames) unique(max_frames)];
    range_offsets = [unique(min_onsets) unique(max_onsets)];
    
    if length(range_offsets) <= 2
        use_offset = min(range_offsets);
        use_nframes = min(range_nframes);
    else
        fprintf('Found > 2 parsing outputs from trial-type parsing:\n');
        fprintf('Range of offset indices: %s\n', mat2str(range_offsets)); 
        fprintf('Range of nframes: %s\n', mat2str(range_nframes));
    end

end

%% PLOT:


for curr_roi=3:3 %size(D.(slice_no).(file_no).raw_traces, 1)
fig = figure('units','normalized','outerposition',[0 0 1 1]);

for curr_stim=1:length(stim_types)
    
    subplot(5,7,curr_stim)
    
    curr_trials = AVG.(stim_types{curr_stim}).trial_deltaF;
    curr_times = AVG.(stim_types{curr_stim}).trial_times;
    
    
    avg_trace = zeros(length(curr_trials), use_nframes);
    avg_time = zeros(length(curr_trials), use_nframes);
    
    
    %onsets = [108]; %1sec baseline
    %onsets = [213]; %[214] %2sec baseline;
    onsets = []; %[214] %2sec no interp;
    for trial=1:length(curr_trials)
        
        curr_trace_idx = find(curr_times{trial}==0);
        onsets = [onsets curr_trace_idx];
        
        if (length(curr_trials{trial}(curr_roi, :)) < length(avg_trace(trial,:)+4))
            fprintf('Missing correct time info for %s, trial %i.\n', stim_types{curr_stim}, trial);
            continue;
        else
                
            if curr_trace_idx == use_offset+1
                avg_trace(trial,:) = curr_trials{trial}(curr_roi, 2:end); %curr_trace_idx-4:17);
                avg_time(trial,:) = curr_times{trial}(2:end); %curr_trace_idx-4:17);

    %         elseif curr_trace_idx == onsets(1)+2
    %             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 3:end);
    %             avg_time(trial,:) = curr_times{trial}(3:end);

            elseif curr_trace_idx == use_offset-1
                %fprintf('Missing correct time info for %s, trial %i.\n', stim_types{curr_stim}, trial);
                %continue;
    %             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 1:end-1);
    %             avg_time(trial,:) = curr_times{trial}(1:end-1);
                avg_trace(trial,:) = curr_trials{trial}(curr_roi, 1:end);
                avg_time(trial,:) = curr_times{trial}(1:end);

    %         elseif curr_trace_idx == onsets(1)-2
    %             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 1:end-1);
    %             avg_time(trial,:) = curr_times{trial}(1:end-1);

            else
    %             if length(curr_trials{trial}(curr_roi, :)) < length(avg_trace(trial,:)+5)
    %                 fprintf('Missing correct time info for %s, trial %i.\n', stim_types{curr_stim}, trial);
    %                 continue;
    %             else
                    avg_trace(trial,:) = curr_trials{trial}(curr_roi, :);
                    avg_time(trial,:) = curr_times{trial};
    %             end
            end
        end
        
        patchline(avg_time(trial,:), avg_trace(trial,:), 'edgecolor','k','linewidth',0.01,'edgealpha',0.5)
        hold on;
    end
    
    hold on;
    plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
    %ylim([-0.3 0.3])
    y1=get(gca,'ylim');
    hold on;

    sy = [y1(1) y1(1) y1(2) y1(2)];
    sx = [0 1 1 0];
    curr_code = stim_types{curr_stim};
    patch(sx, sy, colors(curr_stim,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
    hold on;
    text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_code));

    curr_stim_config = S.gratings(curr_stim,:);
    curr_stim_config_name = sprintf('code%i - ORI: %0.2f, SF: %0.2f', curr_stim, curr_stim_config(1), curr_stim_config(2));
    plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
    title(curr_stim_config_name)
    
    
end

%suptitle(sprintf('ROI %i', active_cells(curr_roi)))
suptitle(sprintf('ROI %i', curr_roi))


fig_prefix = 'PSTH';
tmp_base_struct_fn = strsplit(struct_fn, '/');
base_struct_fn = tmp_base_struct_fn{end}(1:end-4);
figdir = strcat(source_dir, 'figures/');
if ~exist(figdir, 'dir')
    mkdir(figdir)
end
figname = strcat(figdir, sprintf('%s_roi%i_%s.png', fig_prefix, curr_roi, base_struct_fn))
% set(gcf,'units','normalized','outerposition',[0 0 1 1])
%set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig, 'PaperPositionMode', 'auto')
%saveas(fig, char(figname));

end


%%
% 
% for curr_roi=3:3 %size(D.(slice_no).(file_no).raw_traces, 1)
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
% 
% %curr_roi = 2;
% for curr_stim=1:length(stim_types)
% %     if ~isfield(AVG, stim_types{curr_stim})
% %         continue;
% %     end
%     
%     subplot(5,7,curr_stim)
%     
%     curr_trials = AVG.(stim_types{curr_stim}).trial_deltaF;
%     curr_times = AVG.(stim_types{curr_stim}).trial_times;
%     
%     %avg_trace = zeros(length(curr_trials), size(curr_trials{1},2));
%     %avg_time = zeros(length(curr_trials), size(curr_trials{1},2));
%     
%     %avg_trace = zeros(length(curr_trials), size(AVG.(stim_types{1}).trial_deltaF{1},2));
%     %avg_time = zeros(length(curr_trials), size(AVG.(stim_types{1}).trial_deltaF{1},2));
%     
%     %tlength = 1161; %for1sec baseline
%     %tlength = 1266; %1267; %2sec baseline;
%     tlength = 63; %2sec baseline, no interp (1850 frames)
%     avg_trace = zeros(length(curr_trials), tlength);
%     avg_time = zeros(length(curr_trials), tlength);
%     
%     
%     %onsets = [108]; %1sec baseline
%     %onsets = [213]; %[214] %2sec baseline;
%     onsets = [11]; %[214] %2sec no interp;
%     for trial=1:length(curr_trials)
%         
%         curr_trace_idx = find(curr_times{trial}==0);
%         onsets = [onsets curr_trace_idx];
%         
%         if length(curr_trials{trial}(curr_roi, :)) < length(avg_trace(trial,:)+5)
%             fprintf('Missing correct time info for %s, trial %i.\n', stim_types{curr_stim}, trial);
%             continue;
%         end
%                 
%         if curr_trace_idx == onsets(1)+1
%             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 2:end); %curr_trace_idx-4:17);
%             avg_time(trial,:) = curr_times{trial}(2:end); %curr_trace_idx-4:17);
%             
%         elseif curr_trace_idx == onsets(1)+2
%             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 3:end);
%             avg_time(trial,:) = curr_times{trial}(3:end);
%                 
%         elseif curr_trace_idx == onsets(1)-1
%             %fprintf('Missing correct time info for %s, trial %i.\n', stim_types{curr_stim}, trial);
%             %continue;
%             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 1:end-1);
%             avg_time(trial,:) = curr_times{trial}(1:end-1);
%             
% %         elseif curr_trace_idx == onsets(1)-2
% %             avg_trace(trial,:) = curr_trials{trial}(curr_roi, 1:end-1);
% %             avg_time(trial,:) = curr_times{trial}(1:end-1);
%             
%         else
% %             if length(curr_trials{trial}(curr_roi, :)) < length(avg_trace(trial,:)+5)
% %                 fprintf('Missing correct time info for %s, trial %i.\n', stim_types{curr_stim}, trial);
% %                 continue;
% %             else
%                 avg_trace(trial,:) = curr_trials{trial}(curr_roi, :);
%                 avg_time(trial,:) = curr_times{trial};
% %             end
%         end
%         %plot(avg_time(trial,:), avg_trace(trial,:), 'k', 'LineWidth', 0.001)
%         patchline(avg_time(trial,:), avg_trace(trial,:), 'edgecolor','k','linewidth',0.01,'edgealpha',0.5)
%         hold on;
%     end
%     hold on;
%     plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
%     %ylim([-0.3 0.3])
%     y1=get(gca,'ylim');
%     hold on;
% 
%     sy = [y1(1) y1(1) y1(2) y1(2)];
%     sx = [0 1 1 0];
%     curr_code = stim_types{curr_stim};
%     patch(sx, sy, colors(curr_stim,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
%     hold on;
%     text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_code));
% 
%        
%     curr_stim_config = S.gratings(curr_stim,:);
%     curr_stim_config_name = sprintf('code%i - ORI: %0.2f, SF: %0.2f', curr_stim, curr_stim_config(1), curr_stim_config(2));
%     plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
%     title(curr_stim_config_name)
%     
% end
% 
% %suptitle(sprintf('ROI %i', active_cells(curr_roi)))
% suptitle(sprintf('ROI %i', curr_roi))
% 
% 
% fig_prefix = 'PSTH';
% tmp_base_struct_fn = strsplit(struct_fn, '/');
% base_struct_fn = tmp_base_struct_fn{end}(1:end-4);
% figdir = strcat(source_dir, 'figures/');
% if ~exist(figdir, 'dir')
%     mkdir(figdir)
% end
% figname = strcat(figdir, sprintf('%s_roi%i_%s.png', fig_prefix, curr_roi, base_struct_fn))
% % set(gcf,'units','normalized','outerposition',[0 0 1 1])
% %set(fig,'units','normalized','outerposition',[0 0 1 1]);
% set(fig,'units','normalized','outerposition',[0 0 1 1]);
% set(fig, 'PaperPositionMode', 'auto')
% %saveas(fig, char(figname));
% 
% end



%%

mw_tif_fns = fieldnames(trial_struct.code1);

t_fns = {};
tidx = 1;
for mtidx=2:2:length(mw_tif_fns)
   curr_file = sprintf('file%i', runs_to_use(tidx));
   
   if strfind(mw_tif_fns{mtidx}, curr_file) %, sprintf('tif_mw1%s', curr_file)) 
       tidx_fn = mw_tif_fns{mtidx};
       t_fns{tidx} = tidx_fn;
       tidx = tidx + 1;
   end
end

% T = struct();
% for t_fn = 1:length(t_fns)
%    for code=1:length(stim_types)
%        if isfield(trial_struct.(stim_types{code}), t_fns{t_fn})
%            if isfield(T, t_fns{t_fn})
%                T.(t_fns{t_fn})(end+1) = trial_struct.(stim_types{code}).(t_fns{t_fn}); 
%            else
%                T.(t_fns{t_fn})(1) = trial_struct.(stim_types{code}).(t_fns{t_fn});
%            end
%        else
%            continue
%        end
%    end
% end


%%
% for t_fn=1:length(t_fns)
%     X.(t_fns{t_fn}) = T.(t_fns{t_fn});
%     e1 = arrayfun(@(i) X{i}(1), 1:numel(X));
%     [~, O] = sort(e1);
%     X = X(O);
% end
% 
% 

nruns = length(t_fns); %12;

tiff_dir = strcat(source_dir, 'ch1_slice6/');
parse_dir = strcat(source_dir, 'ch1_slice6_parsed/');
if ~exist(parse_dir, 'dir')
    mkdir(parse_dir)
end

% Choose .mat struct fn:
for code=1:length(stim_types)
    
    stim_dir = char(strcat(parse_dir, stim_types(code), '/'));
    if ~exist(stim_dir, 'dir')
        mkdir(stim_dir)
    end

    
    for t_fn=1:length(t_fns)
        if isfield(trial_struct.(stim_types{code}), t_fns{t_fn})
            curr_tfn = t_fns{t_fn};
            curr_tidxs = trial_struct.(stim_types{code}).(curr_tfn){1};

            fprintf('Current run: %i\n', t_fn)
            tmp_file_no = strsplit(t_fns{run_idx}, 'file');
            run_idx = str2num(tmp_file_no{end});

            curr_tiff = sprintf('fov1_gratings_%05d.tif #2.tif #%i.tif', run_idx, slice_idx);
            curr_tiff_path = strcat(tiff_dir, curr_tiff);

            sframe=1;
            Y = bigread2(strcat(tiff_dir, curr_tiff),sframe);
            %if ~isa(Y,'double');    Y = double(Y);  end
            %
            nframes = size(Y,3);

            curr_frames = Y(:,:,curr_tidxs(1):curr_tidxs(end));
            %fprintf('%i frames - File %s - N idxs: %i\n', size(curr_frames,3), curr_tfn, length(curr_tidxs));
            if size(curr_frames,3) == use_nframes+1
                curr_frames = Y(:,:,curr_tidxs(1)+1:curr_tidxs(end));
            end
            
            parsed_tiff_fn = strcat(curr_tiff, sprintf(' %s code%i.tif', curr_tfn, code));
            try
                tiffWrite(curr_frames, parsed_tiff_fn, stim_dir);
            catch
                pause(60);
                tiffWrite(curr_frames, parsed_tiff_fn, stim_dir) %, 'int16');
            end
        else
            continue;
        end
    end

end



%%
%
%
%
%

figure()
for curr_stim=1:length(stim_types)
    subplot(5,7,curr_stim)
    avg_trace = zeros(length(AVG.(stim_types{curr_stim}).trial_deltaF), 17);
    avg_time = zeros(length(AVG.(stim_types{curr_stim}).trial_deltaF), 17);
    
    for curr_trial=1:length(AVG.(stim_types{curr_stim}).trial_deltaF)
        curr_trace_idx = find(AVG.(stim_types{curr_stim}).trial_times{curr_trial}==0);
        
        avg_trace(curr_trial,:) = AVG.(stim_types{curr_stim}).trial_deltaF{curr_trial}(curr_trace_idx-4:17);
        avg_time(curr_trial,:) = AVG.(stim_types{curr_stim}).trial_times{curr_trial}(curr_trace_idx-4:17);
        plot(avg_time(curr_trial,:), avg_trace(curr_trial,:), 'k', 'LineWidth', 0.5)
        hold on;
    end
    hold on;
    plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
    
    curr_stim_config = S.gratings(curr_stim,:);
    curr_stim_config_name = sprintf('ORI: %0.2f, SF: %0.2f', curr_stim_config(1), curr_stim_config(2));
    plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
    title(curr_stim_config_name)
    
end


D.AVG = AVG;
D.trial_struct = trial_struct;
D.stim_types = stim_types;

struct_fn = strcat(source_dir, tmat_fn)
save(struct_fn, 'D')


%% Plot stimulus times with ROI trace:

figure();

for i=1:size(trace_mat,1) %length(ROIs)
    subplot(3,5,i)
    
    %i=ROIs(r);
    
    %%
    % FILE: rsvp? %102; %74 %73; %71; %67; %64; %54; %23; %11
    % FILE: gratings: fov6_gratings_bluemask_5trials_00003 -- 1%10 %5 %4 %3 %1
    % FILE:  gratings2 --2016/12/19_jr030W_/NMF/ -- high-pass.
    %i =  13 %54 %24 %11
   
    i=3
    plot_all = 1;
    
    if plot_all==1
        %figure();
       
        plot(y_sec, trace_mat(i, :).*100, 'k', 'LineWidth', 1)
        y1=get(gca,'ylim');
        hold on;

        y1=get(gca,'ylim');
        x1=get(gca,'xlim');
        %y1 = [-0.1 0.2]; %get(gca,'ylim');
        title(sprintf('Component %i (calcium DF/F value)',i),'fontsize',16,'fontweight','bold');
        %leg = legend('Raw trace (filtered)','Inferred');
        %set(leg,'FontSize',14,'FontWeight','bold');
        xlim([0 mw_sec(end)+3]) %550]);
        drawnow;

        legend_labels = {};
        l=1;
        hold on;
        for stim=1:2:(length(mw_rel_times)-1)
           sx = [mw_sec(stim) mw_sec(stim+1) mw_sec(stim+1) mw_sec(stim)];
           %sx = [mw_frames(stim) mw_frames(stim+1) mw_frames(stim+1) mw_frames(stim)];
           y1=get(gca,'ylim');
           sy = [y1(1) y1(1) y1(2) y1(2)];
           %sy = [-0.1 -0.1 0.1 0.1].*100;
           curr_code = curr_mw_codes(stim);
           patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
           hold on;
           text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_code));
           %text(sx(1), sy(3)-abs(sy(2))*0.5, num2str(curr_code));
           hold on;
           if stim==1
               legend_labels{1} = {num2str(curr_code)};
           else
            legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
           end
        end
    else
        
        
        ntrials_per_chunk = 40;
        mw_chunk = mw_sec;
        while mod(length(mw_chunk), ntrials_per_chunk) > 0
            mw_chunk = padarray(mw_chunk, [0 1], mw_sec(end), 'post');
        end
        ca_chunk = y_sec;
        while mod(length(y_sec), ntrials_per_chunk) > 0
            ca_chunk = padarray(ca_chunk, [0 1], mw_sec(end), 'post');
        end     
            
        mw_chunks = mat2cell(mw_chunk,1,repmat(ntrials_per_chunk*2, [1 length(mw_chunk)/(ntrials_per_chunk*2)]));
        
        for chunk=1:length(mw_chunks)
            %%
           figure(); %Chunks3-4
           chunk = 1
           
           tmp = abs(y_sec-mw_chunks{chunk}(1)); % Find closest matching Ca trace idx for MW trial
           [idx idx] = min(tmp); %index of closest value
           closest = y_sec(idx); %closest value
           start_chunk = idx;
           tmp = abs(y_sec-mw_chunks{chunk+1}(1)); % Find closest matching of end of last MW trial in chunk
           [idx idx] = min(tmp); %index of closest value
           closest = y_sec(idx); %closest value
           end_chunk = idx;
           
           plot(y_sec(start_chunk:end_chunk),guiC(i,start_chunk:end_chunk)/guiDf(i), 'k', 'linewidth',2);
           title(sprintf('Component %i (calcium DF/F value)',i),'fontsize',16,'fontweight','bold');
           %leg = legend('Raw trace (filtered)','Inferred');
           %set(leg,'FontSize',14,'FontWeight','bold');
           drawnow;
           hold on;
           y1=get(gca,'ylim');
           
           mwidx = (chunk-1)*(ntrials_per_chunk*2)+1;
           for stim=mwidx:2:(mwidx+ntrials_per_chunk*2)-1
               sx = [mw_sec(stim) mw_sec(stim+1) mw_sec(stim+1) mw_sec(stim)];
               sy = [y1(1) y1(1) y1(2) y1(2)];
               curr_code = curr_mw_codes(stim);
               patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
               hold on;
               text(sx(1), sy(1)+abs(sy(1))*0.5, num2str(curr_code));
               hold on;
               if stim==1
                   legend_labels{1} = {num2str(curr_code)};
               else
                legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
               end
           end
           %%
        end
        
    end
    
    %

%     h=legend(legend_labels{1}, 'location', 'northeast')
%     set(h, 'position', [0.95 0.3 0.005 0.01])

    
end

%%



y_frames = y_sec./(curr_mw_filedur/nframes);
mw_frames = mw_sec./(curr_mw_filedur/nframes);

legend_labels = {};
l=1;
hold on;
for stim=1:2:length(mw_rel_times)-1
   %sx = [mw_sec(stim) mw_sec(stim+1) mw_sec(stim+1) mw_sec(stim)];
   sx = [mw_frames(stim) mw_frames(stim+1) mw_frames(stim+1) mw_frames(stim)];
   sy = [y1(1) y1(1) y1(2) y1(2)];
   %sy = [-0.1 -0.1 0.1 0.1].*100;
   curr_code = curr_mw_codes(stim);
   patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
   hold on;
   if stim==1
       legend_labels{1} = {num2str(curr_code)};
   else
    legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
   end
end 
