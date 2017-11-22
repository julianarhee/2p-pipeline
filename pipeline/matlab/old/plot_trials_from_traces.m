
clear all;
clc

%% Set paths for current acquisition:

%

source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
mw_path = strcat(source_dir, 'mw_data/');
pymat_fn = '20161222_JR030W_gratings_10reps_run1.mat';


source_slice_no = 8;
source_file_no = 2;
source_struct_fn = strcat(source_dir, sprintf('slice%i_traces_CC.mat', source_slice_no));
source_struct = load(source_struct_fn);
colors = source_struct.D.(['slice' num2str(source_slice_no)]).(['file' num2str(source_file_no)]).colors;
%

slice_idx = 8;
run_idx = 2;
source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
tiff_dir = strcat(source_dir, 'Corrected/');
slice_no = sprintf('slice%i', slice_idx);

% -------------------------------------------------------------------------
% Set here only for source files to be used for getting masks:
curr_tiff = sprintf('fov1_gratings_10reps_run1_Slice%02d_Channel01_File%03d.tif', slice_idx, run_idx);
curr_tiff_path = strcat(tiff_dir, curr_tiff);

struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')
load(struct_fn)


%% Load time series:
nam = strcat(tiff_dir, curr_tiff);

sframe=1;						% user input: first frame to read (optional, default 1)

Y = bigread2(nam,sframe);
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels


%% Get pymat info:
% 
% % Get MW and ARD info from python .mat file:
% pymat_fn = '20161219_JR030W_grating2'; %'fov1_gratings_10reps_run1.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';
% 
% % TEST ON THIS:
% pymat_fn = '20161219_JR030W_gratings_bluemask_5trials_2.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/mw_data/';
% %
% 
% pymat_fn = '20161219_JR030W_rsvp_bluemask_test_10trials.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001/';
% %%
% pymat_fn = 'fov1_gratings_10reps_run1.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/fov1_gratings1_10reps_run1_Slice07_Channel01/NMF/';
% mw_fidx = 6;
% % NOTES:
% % 18 .tif files total, first 5 tifs are bad_scan_phase and _00001.tif is
% % trash run, ignore this one (numbering is +1 since runs start at
% % _00002.tif).  Motion corrected/processed tifs start at _00007.tif, so
% % File00x numbering starts from File001-File013.
% % midx for pymat arrays should be 6 (first legit tif is 7.tif, plus bad
% % 1.tif run).
% 
% 
% pymat_fn = '20161219_JR030W_grating2.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/NMF/';
% mw_fidx = 1;
% %
% pymat_fn = '20161219_JR030W_rsvp_bluemask_test_10trials.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001/';
% mw_fidx = 1;
% %
% 
% pymat_fn = '20161221_JR030W_rsvp_25reps.mat';
% mw_path = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp/';
% % mw_fidx = 4;
% 
% %

%% Load pymat info:

ntrash = 1; %1; %4;
tif_no = 7; %7; %8;
ntifs = 12; %12;
mc_file_no = 1; %4;
abs_mw_fidx = tif_no - ntrash; %mc_file_no+ntrash

S = load(strcat(mw_path, pymat_fn));

mw_times = S.mw_times_by_file;
offsets = double(S.offsets_by_file);
mw_codes = S.mw_codes_by_file;
mw_file_durs = double(S.mw_file_durs);
ard_times = S.ard_file_trigger_times;

% 
% if strcmp(class(mw_times), 'int64')
%     % GET MW rel times from match_triggers.py: 'rel_times'
%     curr_mw_times = double(mw_times(mw_fidx, :)); 
%     
%     % Get MW codes corresponding to time points:
%     curr_mw_codes = mw_codes(mw_fidx, :); 
% else
%     
%     curr_mw_times = double(mw_times{mw_fidx}); % convert to sec
%     curr_mw_codes = mw_codes{mw_fidx};
% end
% mw_rel_times = ((curr_mw_times - curr_mw_times(1)) + offsets(mw_fidx)); % into seconds
% mw_sec = mw_rel_times/1000000;
% 
% 
% % Get correct time dur to convert frames to time (s):
% curr_mw_filedur = mw_file_durs(mw_fidx)/1000000; % convert to sec
% 
% nframes = size(Y, 3);
% 
% % See if ard-file exists and get trigger times from ard file:
% if isfield(S, 'ard_fn')
%     ard_filedurs = double(S.ard_file_durs);
%     y_sec = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes);
% else
%     y_sec = [0:nframes-1].*(curr_mw_filedur/nframes);
% end

% Get n-stim color-code: -- get 'mw_codes' from match_triggers.py:
nstimuli = 35; %length(unique(curr_mw_codes)-1;
% nstimuli = 12;

% if isfield(D.(slice_no).(file_no), 'colors')
%     colors = D.(slice_no).(file_no).colors;
% else
%     colors = zeros(nstimuli,3);
%     for c=1:nstimuli
%         colors(c,:,:) = rand(1,3);
%     end
%     
% end


%% LOAD ROI traces, if they exist:
% tmat_fn = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/slice7_traces_CC_test.mat';
% load(tmat_fn)
% 
% curr_slice = 7;
% curr_file = 9;
% slice_no = sprintf('slice%i', curr_slice);
% 
% source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/';
% tmat_fn = 'slice13_traces_AVG.mat';
% load(strcat(source_dir, tmat_fn))

curr_slice = 8; %13;
slice_no = sprintf('slice%i', curr_slice);

ntifs = length(fieldnames(source_struct.D.(slice_no)));

% file_no = sprintf('file%i', curr_file);
% 
% traces = D.(slice_no).(file_no).rolltraces; %D.(slice_no).(file_no).traces;
% active_cells = D.(slice_no).(file_no).active_cells;

%% Load ROI time courses:
% NMF = 0;
% if NMF==1
%     %[guiT,guiY_r, guiC, guiDf] = plot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);
%     %plot(y_sec,guiY_r(i,:)/guiDf(i), 'k', 'linewidth',.2); hold all; 
%     %plot(y_sec,guiC(i,:)/guiDf(i), 'k', 'linewidth',2);
%     trace_mat = guiC./guiDf(i);
%     trace_mat = traces(active_cells(:), :);
% else
%     % traces = D.(slice_no).(file_no).rolltraces; %D.(slice_no).(file_no).traces;
%     % active_cells = D.(slice_no).(file_no).active_cells;
%     trace_mat = zeros(length(active_cells), size(traces,2));
%     for acell=1:length(active_cells)
%         trace_mat(acell,:) = traces(active_cells(acell),:);
%     end
% end


%% Sort into trials by MW code:
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

% ndiscard = 0 %5; % absolute num of tif files to ignore (since start of MW file) due to scan phase adj.

acquisitions = fieldnames(D.(slice_no));
trial_struct = struct();
% ndiscard = 5;
% ntrash = 1;

for tif=1:ntifs
    
file_no = sprintf('file%i', tif);

traces = D.(slice_no).(file_no).rolltraces; %D.(slice_no).(file_no).traces;
active_cells = D.(['slice', num2str(source_slice_no)]).(['file' num2str(source_file_no)]).active_cells;


%trial_struct = struct();

%for tif=1:ntifs
    mw_fidx = tif + abs_mw_fidx - 1;
    
    mw_file_no = sprintf('mw%ifile%i', mw_fidx, tif);
    if strcmp(class(mw_times), 'int64')
        curr_mw_times = double(mw_times(mw_fidx, :));       % GET MW rel times from match_triggers.py: 'rel_times'
        curr_mw_codes = mw_codes(mw_fidx, :);               % Get MW codes corresponding to time points:
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
        code_no = sprintf('code%i', curr_mw_codes(trial))
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


%% Use trial_struct & selected ROI mask to plot all reps of each stimulus:


[stim_types,~] = sort_nat(fieldnames(trial_struct));
for curr_stim=1:length(stim_types)
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
                t_pre = curr_tstamps(1)+1; % this is actually two seconds before stim onset of next trial
                t_post = curr_tstamps(end); 
                t_onset = curr_tstamps(2); 
            end

            y2 = y_sec(1):(y_sec(2)-y_sec(1))/20:y_sec(end);                % interpolate resampled ard-trigger values to find closer matches to mw-tstamps

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



%% PLOT each:



plot_all = 0;

AVG = struct();
for curr_stim=1:length(stim_types)
    mw_fns = fieldnames(trial_struct.(stim_types{curr_stim}));
    ign_ids = strfind(mw_fns, 'tif');
    mw_fns = mw_fns(find((cellfun('isempty', ign_ids))));
    AVG.(stim_types{curr_stim}) = struct();
    
%     if plot_all==1
%         figure()
%     end
    trial_idx = 1;
    for mwfn=1:length(mw_fns)
        curr_trials = trial_struct.(stim_types{curr_stim}).(mw_fns{mwfn});
        matching_tiff_fn = strcat('tif_', mw_fns{mwfn});
        
        tmp = strsplit(mw_fns{mwfn}, 'file');
        curr_tiff_file_no = ['file' tmp{2}];
        traces = D.(slice_no).(curr_tiff_file_no).rolltraces; %D.(slice_no).(file_no).traces;
        %active_cells = D.(slice_no).(curr_tiff_file_no).active_cells;
        
        %trace_mat = zeros(length(active_cells), size(traces,2));
        trace_mat = zeros(length(active_cells), size(y2,2));
        for acell=1:length(active_cells)
            y_sec2 = interp1(y_sec, traces(active_cells(acell),:), y2); % y2 is interpolated for 20x finer sampling
            trace_mat(acell,:) = y_sec2;
            %trace_mat(acell,:) = traces(active_cells(acell),:);
        end

%         trial_colors = zeros(length(curr_trials),3);
%         for c=1:length(curr_trials)
%             trial_colors(c,:,:) = rand(1,3);
%         end

        for curr_trial=1:length(curr_trials)

            y_start = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(1);
            y_end = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(end);
            y_onset = trial_struct.(stim_types{curr_stim}).(matching_tiff_fn){curr_trial}(2);
            %centered_y = y_sec(y_start:y_end) - y_sec(y_onset);
            centered_y = y2(y_start:y_end) - y2(y_onset);
            
%             % Tweak if needed:
%             closest_pre = find(abs(centered_y+1)==min(abs(centered_y+1))); % Get corresponding frame-time for "1 sec pre"              
%             y_start = y_start + closest_pre - 1;
            while centered_y(end) > 3
                y_end = y_end - 1;
                centered_y = y2(y_start:y_end) - y2(y_onset);
            end
            if length(centered_y) < 300 %16
                fprintf('STIM: %s, MW file: %s, trial: %i\n', stim_types{curr_stim}, mw_fns{mwfn}, curr_trial);
%                 closest_end = y_sec(y_end)+2; % add 2 sec if a cut-off trial (last trial of file)
%                 closest_end = find(abs(y_sec - closest_end) == min(abs(y_sec - closest_end)));
                closest_end = y2(y_end)+2; % add 2 sec if a cut-off trial (last trial of file)
%                 closest_end = find(abs(y2 - closest_end) == min(abs(y2 - closest_end)));
                y_end = closest_end;
                continue;
            end
% %             centered_y = y_sec(y_start:y_end) - y_sec(y_onset);            % Crop mw-indexed times to only include 1 sec before (rather than 2 sec)
            centered_y = y2(y_start:y_end) - y2(y_onset);            % Crop mw-indexed times to only include 1 sec before (rather than 2 sec)

            
%             % Just ignore bad cut-offs for now...
%             trace_y = trace_mat(y_start:y_end);
%             if length(trace_y) < 10
%                 continue
%             end

            
            % Get actual trace values from EACH roi:
            traces_y = NaN(size(active_cells,1), length(centered_y));
            deltaFs_y = NaN(size(active_cells,1), length(centered_y));
            for active=1:length(active_cells)
                if length(centered_y) < 10
                    continue
                end
                traces_y(active,:) = trace_mat(active, y_start:y_end);
                baseline = mean(trace_mat(active, y_start:y_onset-1));
                deltaFs_y(active,:) = ((trace_mat(active, y_start:y_end) - baseline)./baseline) * 100;
            end
            
%             if plot_all == 1
%                 subplot(1,2,1)
%                 plot(centered_y, trace_y)
%                 y1=get(gca,'ylim');
%                 title('raw')
%                 hold on;
%                 
% 
%                 if trial==length(curr_trials)
%                     centered_mw = curr_trials{trial} - curr_trials{trial}(2);
%                     sx = [centered_mw(2) centered_mw(3) centered_mw(3) centered_mw(2)];
%                     sy = [y1(1) y1(1) y1(2) y1(2)];
%                     patch(sx, sy, colors(curr_stim,:,:), 'FaceAlpha', 0.1, 'EdgeAlpha', 0)
%                     hold on;
%                 %if trial==length(curr_trials)
%                     text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_stim));
%                     hold on;
%                 end
%             end

%             baseline = mean(trace_mat(y_start:y_onset-1));
%             deltaF_y = ((trace_mat(y_start:y_end) - baseline)./baseline) * 100;
%             if plot_all==1
%                 subplot(1,2,2)
%                 plot(centered_y, deltaF_y)
%                 hold on;
%             end
            
            AVG.(stim_types{curr_stim}).trial_times{trial_idx} = centered_y; %centered_y;
            AVG.(stim_types{curr_stim}).trial_deltaF{trial_idx} = deltaFs_y;
            AVG.(stim_types{curr_stim}).trial_traces{trial_idx} = traces_y;
            trial_idx = trial_idx + 1;
        end
        
    end
%     curr_stim_config = S.gratings(curr_stim,:);
%     curr_stim_config_name = sprintf('ORI: %0.2f, SF: %0.2f', curr_stim_config(1), curr_stim_config(2));
%     suptitle(curr_stim_config_name)
%     
end


%%  sanity check,

tif = 2;
file_no = sprintf('file%i', tif);
traces = D.(slice_no).(file_no).rolltraces; %D.(slice_no).(file_no).traces;
active_cells = D.(['slice', num2str(source_slice_no)]).(['file' num2str(source_file_no)]).active_cells;

active_idx = 1;
roi_idx = 2;
fidx = 1;

figure()
t_codes = D_mw_codes(1:2:end);

t_trace= [];
for curr_stim=1:length(t_codes)-1
    %t_codes = D_mw_codes(1:2:end);
    pidx = double(t_codes(curr_stim))
    
    subplot(5,4,curr_stim)
    
    t_trace = [t_trace AVG.(stim_types{t_codes(curr_stim+1)}).trial_traces{fidx}(active_idx,:)];
    
    roi_trace = AVG.(stim_types{t_codes(curr_stim+1)}).trial_traces{fidx}(active_idx,:);
    roi_dF = AVG.(stim_types{t_codes(curr_stim+1)}).trial_deltaF{fidx}(active_idx,:);
    roi_times = AVG.(stim_types{t_codes(curr_stim+1)}).trial_times{fidx};
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
        curr_code = t_codes(curr_stim+1);
       patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
       hold on;
       text(sx(1), sy(1)+abs(sy(1))*0.25, num2str(curr_code));

        title(sprintf('code: %i', t_codes(curr_stim+1)))
    end
end

%%

figure()
curr_roi = 2;
for curr_stim=1:length(stim_types)
    
    subplot(5,7,curr_stim)
    
    avg_trace = zeros(length(AVG.(stim_types{curr_stim}).trial_deltaF), 17);
    avg_time = zeros(length(AVG.(stim_types{curr_stim}).trial_deltaF), 17);
    
    for curr_trial=1:length(AVG.(stim_types{curr_stim}).trial_deltaF)
        
        curr_trace_idx = find(AVG.(stim_types{curr_stim}).trial_times{curr_trial}==0);
        
        avg_trace(curr_trial,:) = AVG.(stim_types{curr_stim}).trial_deltaF{curr_trial}(curr_roi, curr_trace_idx-4:17);
        avg_time(curr_trial,:) = AVG.(stim_types{curr_stim}).trial_times{curr_trial}(curr_trace_idx-4:17);
        plot(avg_time(curr_trial,:), avg_trace(curr_trial,:), 'k', 'LineWidth', 0.5)
        hold on;
    end
    hold on;
    plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
    
    curr_stim_config = S.gratings(curr_stim,:);
    curr_stim_config_name = sprintf('code%i - ORI: %0.2f, SF: %0.2f', curr_stim, curr_stim_config(1), curr_stim_config(2));
    plot(mean(avg_time), mean(avg_trace), 'k', 'LineWidth', 2)
    title(curr_stim_config_name)
    
end

suptitle(sprintf('ROI %i', active_cells(curr_roi)))


%%

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
