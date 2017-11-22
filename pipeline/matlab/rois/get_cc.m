

slice_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/fov6_rsvp_nomask_test_10trials_00002/ch1_slices/';
slice_no = 14;
slices = dir(strcat(slice_dir, '/*.tif'));
for i=1:length(slices)
    if findstr(strcat('#',num2str(slice_no),'.tif'), slices(i).name)
        curr_slice = slices(i);
    end
end
curr_slice_path = strcat(slice_dir, curr_slice.name);


curr_slice_source = '/media/juliana/Seagate Backup Plus Drive/RESDATA/20161218_CE024_highres/posterior1/posterior1_4/CH1/';
curr_slice_name = 'posterior1_Slice19_Channel01_File001.tif';
curr_slice_path = strcat(curr_slice_source, curr_slice_name);

%% Get CC images for each slice and run:

% tiff_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_gratings1/Corrected/';
tiff_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_rsvp1/Corrected/';

traces_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_rsvp1/CC/';
tiffs = dir(strcat(tiff_path, '*Channel01*'));

%%
% OLD TEFO.

traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/fov2_bar5_00002/CC/';
tiff_path = '/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/fov2_bar5_00002/ch1_slices/';
curr_tiff_path = strcat(tiff_path, 'fov2_bar5_00002.tif #1.tif #10.tif');
curr_tiff = 'fov2_bar5_00002.tif #1.tif #10.tif'
fidx = 10;
ridx = 2;

%%
% DEFINE SOURCE:
% traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/CC/';
% tiff_path = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/run1_slices/';
% curr_tiff_path = strcat(tiff_path, 'fov3_gratings_00001.tif #2.tif #13.tif');

% STIM:
% fidx = 13;
% ridx = 1;
% tiff_path = sprintf('/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/run%i_slices/', ridx);
% curr_tiff_path = strcat(tiff_path, sprintf('fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx));
% % curr_tiff_path = strcat(tiff_path, sprintf('fov3_gratings_0000%i.tif #2.tif #%i.tif range.tif', ridx, fidx));
% % curr_tiff = sprintf('fov3_gratings_0000%i.tif #2 #%i.tif', ridx, fidx);
% curr_tiff = sprintf('fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);

% BLANK:
% fidx = 13;
% ridx = 1;
% tiff_path = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/blank/ch2_slices/';
% curr_tiff_path = strcat(tiff_path, sprintf('fov3_blank_0000%i.tif #2.tif #%i.tif', ridx, fidx));
% curr_tiff = sprintf('fov3_blank_0000%i.tif #2.tif #%i.tif', ridx, fidx);


%% lower noise? -- lowest on femto

% DEFINE SOURCE:
% run_idx = 6; %10;
% slice_idx = 19;
% traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/CC/';
% tiff_path = sprintf('/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/run%i_slices/', run_idx);
% curr_tiff = sprintf('fov1_gratings_000%02d.tif #2.tif #%i.tif', run_idx, slice_idx);
% curr_tiff_path = strcat(tiff_path, curr_tiff);

%% Check old TEFO, GCaMP

% 
% run_idx = 3; %10;
% slice_idx = 13;
% %traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/CC/';
% tiff_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/ch1_slices/';
% curr_tiff = sprintf('fov6_gratings_bluemask_5trials_0000%i.tif #1 #%i.tif', run_idx, slice_idx);
% curr_tiff_path = strcat(tiff_path, curr_tiff);
% 

%% Check 12k, 2x arm:


run_idx = 2; %10;
slice_idx = 8;

source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
tiff_dir = strcat(source_dir, 'Corrected/');

curr_tiff = sprintf('fov1_gratings_10reps_run1_Slice%02d_Channel01_File%03d.tif', slice_idx, run_idx);
curr_tiff_path = strcat(tiff_dir, curr_tiff);

avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);

slice_no = ['slice' num2str(slice_idx)];
file_no = ['file' num2str(run_idx)]; % 'file1R'

parts = strsplit(source_dir, '/');
cond_folder = parts{end-1};


sframe=1;

for tiff=1:length(tiffs)
    curr_tiff = tiffs(tiff).name;
    curr_tiff_path = strcat(tiff_path, curr_tiff);

    %%
    Y = bigread2(curr_tiff_path,sframe);

    %Y = Y - min(Y(:)); 
    if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

    [d1,d2,T] = size(Y);                                % dimensions of dataset
    d = d1*d2;                                          % total number of pixels

    [cc]=CrossCorrImage(Y);
    cc_fn = strcat(traces_path, 'cc_', curr_tiff(1:end-4), '.png');
    imwrite(cc, cc_fn);
end


%%
% 
% slice_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_gratings1/slice10/';
% avgimg_fn = 'AVG_fov1_gratings_10reps_run1_Slice10_Channel01_File001_scaled.tif';
% tseries_fn = 'fov1_gratings_10reps_run1_Slice10_Channel01_File001_scaled.tif';

% slice_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_gratings1/fov1_gratings_10reps_run1_slice6_00009/'
% avgimg_fn = 'AVG_fov1_gratings_10reps_run1_Slice06_Channel01_File009_scaled.tif';
% corrected_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_gratings1/Corrected/';
% tseries_fn = 'fov1_gratings_10reps_run1_Slice06_Channel01_File009_scaled.tif';

% source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
% slice_dir = strcat(source_dir, 'fov1_gratings1_10reps_run1_Slice07_Channel01/');
% 
% slice_path = strcat(source_dir, 'fov1_gratings1_10reps_run1_Slice07_Channel01_File002/');
% avgimg_fn = 'AVG_fov1_gratings_10reps_run1_Slice07_Channel01_File002_scaled.tif';
% ccimg_fn = 'cc_fov1_gratings_10reps_run1_Slice07_Channel01_File002_scaled.tif';
% corrected_path = '/nas/volume1/2photon/RESDATA/20161222_JR030W_gratings1/Corrected/';

%
% fidx = 9;
% slice_dir = strcat(source_dir, 'fov1_gratings1_10reps_run1_Slice07_Channel01/');
% tseries_fn = sprintf('fov1_gratings_10reps_run1_Slice07_Channel01_File0%02d_scaled.tif', fidx);
% avgimg_fn = sprintf('AVG_fov1_gratings_10reps_run1_Slice07_Channel01_File0%02d_scaled.tif', fidx);
% ccimg_fn = sprintf('cc_fov1_gratings_10reps_run1_Slice07_Channel01_File0%02d_scaled.tif', fidx);

%
%fidx = 13;
% source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/';
% slice_dir = strcat(source_dir, 'ch1_slices/');
% tseries_fn = sprintf('fov6_gratings_bluemask_5trials_00003.tif #1 #%i.tif', fidx);
% avgimg_fn = sprintf('AVG_fov6_gratings_bluemask_5trials_00003_Channel01_Slice%i.tif', fidx);

%% specify im dirs for finding ROIs:

% fidx=13;
% ridx = 1;
%source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/';
%source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/fov2_bar5_00002/';
%source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/';
% source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/gratings/fov6_gratings_bluemask_5trials_00003/';
% use_blank = 0;
% 
% if use_blank==1
%     slice_dir = strcat(source_dir, 'blank/ch2_slices/');
%     tseries_fn = sprintf('fov3_blank_0000%i.tif #2.tif #%i.tif', ridx, fidx);
%     avgimg_fn = sprintf('blank/AVG_fov3_blank_0000%i.tif #2.tif #%i.tif', ridx, fidx);
%     ccimg_fn = strcat('CC/', sprintf('cc_fov3_blank_0000%i.tif #2.tif #%i.png', ridx, fidx));
% else
%     %slice_dir = strcat(source_dir, sprintf('run%i_slices/', run_idx));
%     slice_dir = strcat(source_dir, 'ch1_slices/');
%     tseries_fn = curr_tiff; %sprintf('fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);
%     avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
%     ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
%     stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);
% end
% 
% slice_no = ['slice' num2str(slice_idx)];
% file_no = ['file' num2str(run_idx)]; % 'file1R'
% 
% parts = strsplit(source_dir, '/');
% cond_folder = parts{end-1};

%%
% Choose .mat struct fn:
% struct_fn = strcat(source_dir, 'CC_', slice_no, sprintf('_traces_%s.mat', cond_folder));

struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')
% struct_fn = strcat(source_dir, slice_no, '_traces_STD.mat')
% struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')


% get average:
avgimg = imread(strcat(source_dir, avgimg_fn));
avgimg = mat2gray(avgimg);
% imshow(avgimg)

% get CC img to find active ROIs:
ccimg = imread(strcat(source_dir, ccimg_fn));
ccimg = mat2gray(ccimg);
% imshow(ccimg)

% get STD:
stdimg = imread(strcat(source_dir, stdimg_fn));
stdimg = mat2gray(stdimg);
% imshow(avgimg)

% choose ROIs:
% masks=ROIselect_circle(stdimg);
masks=ROIselect_circle(ccimg);
% masks=ROIselect_circle(avgimg);


if ~exist(struct_fn)
    D = struct;
    D.(slice_no) = struct();
    D.(slice_no).(file_no) = struct();

    D.(slice_no).(file_no).masks = masks;
    D.(slice_no).(file_no).avgimg = avgimg;
    D.(slice_no).(file_no).ccimg = ccimg;
    D.(slice_no).(file_no).curr_tiff = curr_tiff;
    D.(slice_no).(file_no).slice_path = tiff_dir;
else
    load(struct_fn);
    if ~isfield(D.(slice_no), file_no)
        display('Current slice exists, but not current file.')
        D.(slice_no).(file_no) = struct();
%     else
%         load(struct_fn);
    end
end
% struct_fn = strcat(source_dir, slice_no, '_traces_AVG.mat')
% save(struct_fn, 'D')

%% Load masks & relevant struct info if using previously-defind ROIs:

% if needed, load masks to use:
%struct_fn = strcat(source_dir, slice_no, '_traces_AVG.mat');
struct_fn = strcat(source_dir, slice_no, '_traces_AVG_run6_smallroi.mat');

load(struct_fn)
masks = D.slice13.file1.masks;

% load movie:
sframe=1;
Y = bigread2(strcat(slice_dir, tseries_fn),sframe);
if ~isa(Y,'double');    Y = double(Y);  end

%%

nframes = size(Y,3);

% extract raw traces:
raw_traces = zeros(size(masks,3), size(Y,3));
for r=1:size(masks,3)
    curr_mask = masks(:,:,r);
    Y_masked = zeros(1,size(Y,3));
    for t=1:size(Y,3)
        t_masked = curr_mask.*Y(:,:,t);
        Y_masked(t) = sum(t_masked(:));
    end
    raw_traces(r,:,:) = Y_masked;
end

figure()
for rtrace=1:size(raw_traces,1)
    plot(raw_traces(rtrace,:), 'color', rand(1,3))
    hold on;
end

if isfield(D.(slice_no).(file_no), 'raw_traces')
    display(sprintf('Traces for file %s already exists.', file_no))
    continue;
else
    D.(slice_no).(file_no).raw_traces = raw_traces;
    save(struct_fn, 'D')
end

%% Processes traces.

% rolling avg
% frameArray = reshape(Y, d, size(Y,3));
% framesToAvg = 104;
% for pix = 1:length(frameArray)
%     tmp0=frameArray(pix,:);
%     tmp1=padarray(tmp0,[0 framesToAvg],tmp0(1),'pre');
%     tmp1=padarray(tmp1,[0 framesToAvg],tmp0(end),'post');
%     rollingAvg=conv(tmp1,fspecial('average',[1 framesToAvg]),'same');%average
%     rollingAvg=rollingAvg(framesToAvg+1:end-framesToAvg);
%     frameArray(pix,:) = tmp0 - rollingAvg;
% end
% rollY = reshape(frameArray, size(Y));

% winsize = 100;
winsize = 104;
winsize = 156; % 3 trials

roll_winsize = 25; %104; % 2 trials
% rolling avg for each trace:
rolltraces = zeros(size(raw_traces));
for rtrace=1:size(raw_traces,1)
    tmp0 = raw_traces(rtrace,:);
    tmp1=padarray(tmp0,[0 roll_winsize],tmp0(1),'pre');
    tmp1=padarray(tmp1,[0 roll_winsize],tmp0(end),'post');
    rollingAvg=conv(tmp1,fspecial('average',[1 roll_winsize]),'same');%average
    rollingAvg=rollingAvg(roll_winsize+1:end-roll_winsize);
    rolltraces(rtrace,:) = tmp0 - rollingAvg;
end

% high-pass filter traces:
winsize = 25; %200; %185;

deltaF = zeros(size(raw_traces));
traces = zeros(size(raw_traces));
for rtrace=1:size(raw_traces,1)
    
    curr_trace = raw_traces(rtrace,:);
    s1 = smooth(curr_trace, winsize, 'rlowess');
    t1 = curr_trace - s1';

    baseline = mean(t1);
    dF = (t1 - baseline)./baseline; 

%     figure()
%     subplot(1,2,1)
%     plot(curr_trace, 'k')
%     hold on
%     plot(s1, 'r')
%     subplot(1,2,2)
%     plot(t1, 'b')
%     figure()
%     plot(dF, 'g');
%         
    deltaF(rtrace,:) = dF;
    traces(rtrace,:) = t1;
end

D.(slice_no).(file_no).winsize = winsize;
D.(slice_no).(file_no).traces = traces;
D.(slice_no).(file_no).rolltraces = rolltraces;
D.(slice_no).(file_no).roll_winsize = roll_winsize;
D.(slice_no).(file_no).deltaF = deltaF;

save(struct_fn, 'D')

%% Find active cells:

% Choose cutoff by max peak:
peak_threshold = 0.05; %0.6 %0.05;
peak_cutoff = 100 %100; %30; %200;
active_cells = [];
figure()
for rtrace=1:size(traces,1)
    %check_max = (traces(rtrace,:) - mean(traces(rtrace,:)) ) ./ mean(traces(rtrace,:));
    %check_max = (traces(rtrace,:) - mean(traces(rtrace,:)) ) ./ mean(traces(rtrace,:));
    %if max(check_max) > peak_threshold
    %if (peak_cutoff > max(check_max) > peak_threshold)
    if (peak_cutoff > max(deltaF(rtrace,:)) > peak_threshold)
    %if (max(deltaF(rtrace,:)) > peak_threshold)
    %if (max(traces(rtrace,:)) > peak_threshold)
        active_cells = [active_cells rtrace];
        plot(traces(rtrace,:), 'color', rand(1,3))
        hold on;
    end
end
title(sprintf('%i ROIs found above threshold %i%%', length(active_cells), (peak_threshold*100)))

D.(slice_no).(file_no).active_cells = active_cells;
D.(slice_no).(file_no).peak_threshold = peak_threshold;
D.(slice_no).(file_no).peak_cutoff = peak_cutoff;
save(struct_fn, 'D')


%% Match traces to stimuli:

% Get MW and ARD info from python .mat file:
% pymat_fn = 'fov1_gratings_10reps_run1.mat';
% 
% pymat_fn = '20161219_JR030W_gratings_bluemask_5trials_2.mat';
% S = load(strcat(source_dir, 'mw_data/', pymat_fn));

pymat_fn = '20161222_JR030W_gratings_10reps_run1.mat';
S = load(strcat(source_dir, 'mw_data/', pymat_fn));

%pymat_fn = sprintf('20160115_AG33_gratings_fov3_run%i.mat', ridx);
%pymat_fn = sprintf('20160118_AG33_gratings_fov1_run%i.mat', run_idx);
%S = load(strcat(source_dir, pymat_fn));

ard_filedurs = double(S.ard_file_durs);
mw_times = S.mw_times_by_file;
offsets = double(S.offsets_by_file);
mw_codes = S.mw_codes_by_file;
mw_filedurs = double(S.mw_file_durs);

%% colormap

% first_struct_slice = 'slice11';
% first_struct_file = 'file1';
% first_struct = load(strcat(source_dir, 'slice11_traces_AVG.mat'));
% 
% first_struct_slice = 'slice13';
% first_struct_file = 'file1';
% first_struct = load(strcat(source_dir, 'CC_slice13_traces_fov6_gratings_bluemask_5trials_00003.mat'));
% colors = first_struct.D.(first_struct_slice).(first_struct_file).colors;


first_struct_slice = 'slice7';
first_struct_file = 'file2';
first_struct = load(strcat(source_dir, 'slice7_traces_cc2.mat'));
colors = first_struct.D.(first_struct_slice).(first_struct_file).colors;


% Get n-stim color-code: -- get 'mw_codes' from match_triggers.py:
nstimuli = 35; %length(unique(mw_codes(1,:)))-1;
colors = zeros(nstimuli,3);
for c=1:nstimuli
    colors(c,:,:) = rand(1,3);
end
% figure();
% for c=1:nstimuli
%    plot([1], [c], '.', 'MarkerSize', 10, 'MarkerFaceColor',  colors(c,:,:))
%    xlim([0,2]);
%    hold on;
% end

%% Stimulus:

% GET ard_file_duration for current file: _0000x.tif
mw_start_tif = 2; % if acquis. file incrementer not re-started (i.e,. threw out _00001.tif)
ndiscard = 6; % removed previous _0000x.tifs due to scan phase adj (12/21-12/22)
mw_fidx = 1%fidx + ndiscard
   
% 
% if strcmp(file_no, 'file13')
%     y_sec1 = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes);
%     y_sec1 = y_sec1(1:2:end);
%     y_int = diff(y_sec1);
%     y_int = y_int(1);
%     y_sec2 = y_sec1 + y_sec1(end);
%     y_sec = [y_sec1 y_sec2(2:end) y_sec2(end)+y_int];
% else
%     y_sec = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes); % y_sec = [0:nframes-1]./acquisition_rate;
%     % ^^ this doesn't work for tifs where mw ends before end of file
% end


% ntrash = 4;
% tif_no = 8;
% mc_file_no = 4;
% mw_fidx = mc_file_no+ntrash;

ntrash = 1;
tif_no = 7;
mc_file_no = 1;
mw_fidx = tif_no - ntrash; %mc_file_no+ntrash;

if strcmp(class(mw_times), 'int64')
    % GET MW rel times from match_triggers.py: 'rel_times'
    curr_mw_times = double(mw_times(mw_fidx, :)); 
    
    % Get MW codes corresponding to time points:
    curr_mw_codes = mw_codes(mw_fidx, :); 
else
    curr_mw_times = double(mw_times{mw_fidx}); % convert to sec
    curr_mw_codes = mw_codes{mw_fidx};
end
mw_rel_times = ((curr_mw_times - curr_mw_times(1)) + offsets(mw_fidx)); % into seconds
mw_sec = mw_rel_times/1000000;


% Get correct time dur to convert frames to time (s):
curr_mw_filedur = mw_filedurs(mw_fidx)/1000000; % convert to sec

nframes = size(Y, 3);

if ~exist(ard_filedurs)
    y_sec = [0:nframes-1].*(curr_mw_filedur/nframes);
else
    y_sec = [0:nframes-1].*(ard_filedurs(mw_fidx)/nframes);
end
    
    

% save, if needed:
D.(slice_no).(file_no).mw_sec = mw_sec;
D.(slice_no).(file_no).mw_codes = curr_mw_codes;
D.(slice_no).(file_no).si_to_sec = y_sec;
D.(slice_no).(file_no).nframes = nframes;
D.(slice_no).(file_no).ard_filedur = ard_filedurs(mw_fidx);%
D.(slice_no).(file_no).offset = offsets(mw_fidx);
D.(slice_no).(file_no).mw_times = curr_mw_times;
D.(slice_no).(file_no).mw_rel_times = mw_rel_times;

save(struct_fn, 'D')

%%  PLOT traces with color-coded stimulus epochs:

D.(slice_no).(file_no).colors = colors;
save(struct_fn, 'D')

length(active_cells)
%%
curr_slice = 8; %13; %19; %11;
curr_run = 2; %3; %6; %10; %1;
no_mw = 0;

%D_traces = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).deltaF;
%D_traces = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).traces;
D_traces = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).rolltraces;
D_active = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).active_cells;

if curr_run==0 || no_mw==1
    D_ysec = [1:length(D_traces)]; 
else
    D_ysec = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).si_to_sec;
    D_mw_rel_times = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).mw_rel_times;
    D_mw_sec = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).mw_sec;
    D_mw_codes = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).mw_codes;
end


%%
figure();
for pidx=1:length(active_cells)+1
   %subplot(length(active_cells), 1, cell)
   subplot(5,2,pidx)
   
   %%
   cell = 2; %pidx; %pidx % + 30;
   
   figure()
   %
   
  % plot trace:
   %plot(y_sec, traces(active_cells(cell), :).*100, 'k', 'LineWidth', 1)
   if cell < length(D_active)+1
        plot(D_ysec, D_traces(D_active(cell), :), 'k', 'LineWidth', 1)
   else
       plot(D_ysec, D_traces(D_active(cell-1), :), 'w', 'LineWidth', 1)
   end
   y1=get(gca,'ylim');
   hold on;
   
   %plot(mw_rel_times, zeros(size(mw_rel_times)), 'r*'); 
   legend_labels = {};
   l=1;
   for stim=1:2:length(D_mw_rel_times)-1
       sx = [D_mw_sec(stim) D_mw_sec(stim+1) D_mw_sec(stim+1) D_mw_sec(stim)];
       sy = [y1(1) y1(1) y1(2) y1(2)];
       %sy = [-0.1 -0.1 0.1 0.1].*100;
       curr_code = D_mw_codes(stim);
       patch(sx, sy, colors(curr_code,:,:), 'FaceAlpha', 0.5, 'EdgeAlpha', 0)
       hold on;
       if cell == length(D_active)+1
        %text(sx(1), sy(1)+abs(sy(1))*0.5, num2str(curr_code));
        text(sx(1), sy(3)-abs(sy(2))*0.5, num2str(curr_code));
        hold on;
       end
       if stim==1
           legend_labels{1} = {num2str(curr_code)};
       else
        legend_labels{1} = [legend_labels{1} {num2str(curr_code)}];
       end
   end

   
   
   % plot trace:
   %plot(y_sec, traces(active_cells(cell), :).*100, 'k', 'LineWidth', 2)
   if cell==length(D_active)+1
       title('Stim ID')
   else
       title(['ROI #: ' num2str(D_active(cell))]);
   end
   hold on;
   
   %
end

%%
h=legend(legend_labels{1}, 'location', 'northeast')
set(h, 'position', [0.95 0.6 0.05 0.2])
xlabel('time (s)')
% ylabel('delta F / F (%)')

h = axes('Position',[0 0 1 1],'Visible','off'); %add an axes on the left side of your subplots
set(gcf,'CurrentAxes',h)
text(.1,.45,'delta F / F (%)',...
'VerticalAlignment','bottom',...
'HorizontalAlignment','left', 'Rotation', 90, 'FontSize',18)

% h = axes('Position',[1 1 0 0],'Visible','off'); %add an axes on the left side of your subplots
% set(gcf,'CurrentAxes',h)
% text(.1,.45,'time (s)',...
% 'VerticalAlignment','bottom',...
% 'HorizontalAlignment','center', 'Rotation', 0, 'FontSize',18)

D.(slice_no).(file_no).colors = colors;
D.(slice_no).(file_no).legend = legend_labels{1};
save(struct_fn, 'D')

%% Weird oscillations?

cell = 7 %(fov2_bar5_00002)
S = D_traces(D_active(cell), :);

S = t1; %traces(6,:);

Fs = 5.58;
T = 1/Fs;
L = length(S);
t = (0:L-1)*T;


figure()
plot(t, S, 'k', 'LineWidth', 1)

Y = fft(S);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);


figure()
target_freq = 0.37;
f1 = Fs*(0:(L/2))/L;
plot(f1,P1)
hold on;
plot(target_freq, max(P1), 'r*')
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%%  PHASE MAP?

target_freq = 0.37;
Fs = 5.58;

avgimg = D.(slice_no).(file_no).avgimg;
masks = D.(slice_no).(file_no).masks;

winsz = 30;
for roi=1:size(D.(slice_no).(file_no).raw_traces,1)
    
    roi_no = sprintf('roi%i', roi);
    
    fin = round((5.58/.37)*30);
%     pix = D.(slice_no).(file_no).raw_traces(roi, 1:round(fin));
%     actualT = 0:length(pix);
%     interpT = 0:(1/5.58):fin;
%     tmpy = interp1(actualT, pix, interpT);
    
    tmpy = D.(slice_no).(file_no).raw_traces(roi, 1:round(fin));
    %tmpy = D.(slice_no).(file_no).raw_traces(roi, 1:end);
    tmp1 = padarray(tmpy,[0 winsz],tmpy(1),'pre');
    tmp1 = padarray(tmp1,[0 winsz],tmpy(end),'post');
    rollingAvg=conv(tmp1,fspecial('average',[1 winsz]),'same');%average
    rollingAvg=rollingAvg(winsz+1:end-winsz);
    y = tmpy - rollingAvg;
    
    NFFT = length(y);
    Y = fft(y,NFFT);
    F = ((0:1/NFFT:1-1/NFFT)*Fs).';
    freq_idx = find(abs((F-target_freq))==min(abs(F-target_freq)));

    magY = abs(Y);
    %phaseY = unwrap(angle(Y));
    phaseY = angle(Y);
    %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
    %phase_deg = phase_rad / pi * 180 + 90;
    
    fft_struct.(roi_no).targetPhase = phaseY(freq_idx); % unwrap(angle(Y(freq_idx)));
    fft_struct.(roi_no).targetMag = magY(freq_idx);
    
end

phaseMap = zeros(size(avgimg));
phaseMap(phaseMap==0) = NaN;
for roi=1:length(fieldnames(fft_struct))
    
    roi_no = sprintf('roi%i', roi);
    curr_phase = fft_struct.(roi_no).targetPhase;
    curr_mag = fft_struct.(roi_no).targetMag;
    replaceNan = masks(:,:,roi)==1;
    phaseMap(replaceNan) = 0;
    phaseMap = phaseMap + curr_phase*masks(:,:,roi);
    
end

figure()
A = repmat(avgimg, [1, 1, 3]);
B = phaseMap;
imagesc(A);
hold on;
Bimg = imagesc2(B);
colormap hsv
caxis([-pi, pi])
%colorbar()

%% legend?

sim_path = '/home/juliana/Repositories/retinotopy-mapper/tests/simulation/';
sim_fn = 'V-Left_0_8bit.tif';

sframe = 1;
SIM = bigread2(strcat(sim_path, sim_fn),sframe);
if ~isa(SIM,'double');    SIM = double(SIM);  end
nframes = size(SIM,3);

Fs = 60;
T = 1/Fs;
L = nframes;
t = (0:L-1)*T;
target = 0.05;
legend_phase = zeros(size(SIM,1), size(SIM,2));
for x=1:size(SIM,1)
    for y=1:size(SIM,2)
        pix = SIM(x,y,:);
        NFFT = length(pix);
        Yf = fft(pix,NFFT);
        F = ((0:1/NFFT:1-1/NFFT)*Fs).';
        freq_idx = find(abs((F-target))==min(abs(F-target)));

        %magY = abs(Yf);
        %phaseY = unwrap(angle(Y));
        %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
        %phase_deg = phase_rad / pi * 180 + 90;

        legend_phase(x,y) = angle(Yf(freq_idx)); % unwrap(angle(Y(freq_idx)));
    end
        
end
figure()
imagesc(legend_phase)
colormap hsv
caxis([-pi, pi])
colorbar()


%% fake legend:

A = zeros(10,50);
A(:,1) = ones(size(A(:,1)));
legend_im = zeros(10,50,1000);
legend_im(:,:,1) = A(:,:,1);
tmpA = A;
for lidx=2:1000
    legend_im(:,:,lidx) = circshift(tmpA, 1, 2);
    tmpA = legend_im(:,:,lidx);
end

Fs = 1;
T = 1/Fs;
L = size(legend_im,3);
t = (0:L-1)*T;
target_freq = 1/50;
legend_phase = zeros(size(legend_im,1), size(legend_im,2));
for x=1:size(legend_im,1)
    for y=1:size(legend_im,2)
        pix = legend_im(x,y,:);
        NFFT = length(pix);
        Y = fft(pix,NFFT);
        F = ((0:1/NFFT:1-1/NFFT)*Fs).';
        freq_idx = find(abs((F-target_freq))==min(abs(F-target_freq)));

        magY = abs(Y);
        %phaseY = unwrap(angle(Y));
        %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
        %phase_deg = phase_rad / pi * 180 + 90;

        legend_phase(x,y) = angle(Y(freq_idx)); % unwrap(angle(Y(freq_idx)));
    end
        
end
figure()
imagesc(legend_phase)
colormap hsv
caxis([-pi, pi])
%colorbar()
axis('off')
%helperFrequencyAnalysisPlot1(F, magY, phaseY, NFFT)

%% How many active cells have matching freq?

target_freq = 0.37;
matchingROIs = [];
for roi=1:length(active_cells)
    x = D_traces(D_active(roi), :);
    
    
    psdest = psd(spectrum.periodogram,x,'Fs',Fs,'NFFT',length(x));
    [~,I] = max(psdest.Data);
    fprintf('Maximum occurs at %d Hz.\n',psdest.Frequencies(I));
    
    if sprintf('%0.2f', psdest.Frequencies(I))==num2str(target_freq)
        matchingROIs = [matchingROIs D_active(roi)];
    end

end

fprintf('Found %i out of %i total ROIs with peak at target frequency of %0.2f', length(matchingROIs), length(active_cells), target_freq)



%% Plot ROIs and show active cells:

figure()
% nparts = strsplit(tseries_fn, '_');
% figname = strcat(nparts(1), nparts(2), nparts(3), nparts(4), nparts(5), nparts(7));

nparts = strsplit(tseries_fn, ' ');
figname = strcat(nparts(1), '_', slice_no, '_', file_no);
figname = strrep(figname, '_', '-')

RGBimg = zeros([size(avgimg),3]);
RGBimg(:,:,1)=0;
RGBimg(:,:,2)=avgimg;
RGBimg(:,:,3)=0;

%figure()
numcells=size(D.(slice_no).(file_no).masks,3);
for c=1:numcells
    RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,c);
%     if ismember(c, active_cells)
%         RGBimg(:,:,1)=RGBimg(:,:,1)+0.2*masks(:,:,c);
%     end
end

% Find ROIs that match target freq (only retino):
% for c=1:numcells
%     %RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,c);
%     if ismember(c, matchingROIs)
%         RGBimg(:,:,1)=RGBimg(:,:,1)+0.3*masks(:,:,c);
%     end
% end

imshow(RGBimg);
title(figname);
hold on;
for ac=1:numcells
    if ismember(ac, active_cells)
        [x,y] = find(masks(:,:,ac)==1,1);
        hold on;
        text(y,x,num2str(ac))
    end
end
hold on;

for c=1:numcells
    [x,y] = find(masks(:,:,c)==1,1);
    text(y,x,num2str(c))
    hold on;
end

D.(slice_no).(file_no).RGBimg = RGBimg;
save(struct_fn, 'D')


%%
% save traces and RBG image with ROIs:
D = struct;
D.winsize = winsize;
D.cc_masks = masks;
D.RGB = RGBimg;
D.active = active_cells;
D.traces = traces;
D.raw_traces = raw_traces;
D.avgimg = avgimg;
D.ccimg = ccimg;
D.tseries_fn = tseries_fn;
D.slice_path = slice_path;
D.peak_threshold = peak_threshold;

A = struct();
A.avg_masks = masks;
A.RGB = RGBimg;
A.active = active_cells;
A.traces = traces;
A.raw_traces = raw_traces;
A.avgimg = avgimg;
A.tseries_fn = tseries_fn;
A.slice_path = slice_path;
