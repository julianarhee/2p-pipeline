clear all;
clc;

%% DEFINE SOURCE:

% Check 12k, 2x arm:
% 
% run_idx = 2; %10;
% slice_idx = 8;
% 
% source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/20161222_JR030W_gratings1/';
% tiff_dir = strcat(source_dir, 'Corrected/');
% 
% % -------------------------------------------------------------------------
% % Set here only for source files to be used for getting masks:
% curr_tiff = sprintf('fov1_gratings_10reps_run1_Slice%02d_Channel01_File%03d.tif', slice_idx, run_idx);
% curr_tiff_path = strcat(tiff_dir, curr_tiff);
% 
% avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
% ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
% stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);
% 
% slice_no = ['slice' num2str(slice_idx)];
% file_no = ['file' num2str(run_idx)]; % 'file1R'
% % -------------------------------------------------------------------------
% 
% parts = strsplit(source_dir, '/');
% cond_folder = parts{end-1};
% 
% % Set dir for new (or to-be-appended struct):
% struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')
% % struct_fn = strcat(source_dir, slice_no, '_traces_STD.mat')
% % struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')


%% Define Source:

run_idx = 10;
slice_idx = 1;

source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20170118_AG33/fov1_gratings1/';
tiff_dir = strcat(source_dir, sprintf('run%i_slices/', run_idx));
% source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20170118_AG33/fov1_gratings1/ch1_slice6_parsed/test_avg/';
% tiff_dir = '/nas/volume1/2photon/RESDATA/TEFO/20170118_AG33/fov1_gratings1/ch1_slice6_parsed/test_avg/';

% -------------------------------------------------------------------------
% Set here only for source files to be used for getting masks:
curr_tiff = sprintf('fov1_gratings_%05d.tif #2.tif #%i.tif', run_idx, slice_idx);
curr_tiff_path = strcat(tiff_dir, curr_tiff);


% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

run_idx = 3;
slice_idx = 15;

source_dir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/rsvp_run3/';
tiff_dir = sprintf('Corrected_Slice%i_Channel01/', slice_idx);
% source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20170118_AG33/fov1_gratings1/ch1_slice6_parsed/test_avg/';
% tiff_dir = '/nas/volume1/2photon/RESDATA/TEFO/20170118_AG33/fov1_gratings1/ch1_slice6_parsed/test_avg/';

% -------------------------------------------------------------------------
% Set here only for source files to be used for getting masks:

tiffs = dir(fullfile(source_dir, tiff_dir, '*Channel01*.tif'));
tiffs = {tiffs(:).name}';

fidx = 1 
curr_tiff = sprintf('fov2_rsvp_25reps_run3_Slice%i_Channel01_File%03d.tif', run_idx, slice_idx, fidx);
curr_tiff_path = fullfile(source_dir, tiff_dir, curr_tiff);


%% Load fidx TIF, and create average for masks:
sframe = 1;
Y = bigread2(curr_tiff_path,sframe);

%Y = Y - min(Y(:)); 
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;             

avgY = mean(Y, 3);
imagesc(avgY);


%% Generic info for loading / saving structs based on curr_tiff:

% code_idx = 3;
% curr_tiff = sprintf('AVG_code%i.tif', code_idx);
% curr_tiff_path = strcat(tiff_dir, curr_tiff);
% avgimg_fn = sprintf('MAX_%s', curr_tiff);

avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
%stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);

slice_no = ['slice' num2str(slice_idx)];
file_no = ['file' num2str(fidx)]; % 'file1R'
% file_no = ['code' num2str(code_idx)]; % 'file1R'
% -------------------------------------------------------------------------

parts = strsplit(source_dir, '/');
cond_folder = parts{end-1};

% Set dir for new (or to-be-appended struct):
struct_fn = strcat(source_dir, slice_no, '_traces_AVG.mat')
%struct_fn = strcat(source_dir, slice_no, '_traces_AVG_singleROI_dim.mat')
%struct_fn = strcat(source_dir, slice_no, '_traces_AVGTIFF.mat')

% struct_fn = strcat(source_dir, slice_no, '_traces_STD.mat')
% struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')


%% Get CC images:
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
    cc_fn = strcat(source_dir, 'CC/cc_', curr_tiff(1:end-4), '.png');
    imwrite(cc, cc_fn);
end

%% Get old mask from source-struct, or create new:

make_new_rois = 0;
% 
% % get average:
% avgimg = imread(strcat(source_dir, avgimg_fn));
% avgimg = mat2gray(avgimg);
% % imshow(avgimg)
% 
% % get CC img to find active ROIs:
% ccimg = imread(strcat(source_dir, ccimg_fn));
% ccimg = mat2gray(ccimg);
% % imshow(ccimg)

% get STD:
% stdimg = imread(strcat(source_dir, stdimg_fn));
% stdimg = mat2gray(stdimg);
% imshow(avgimg)

% choose ROIs:

if make_new_rois == 1
    % masks=ROIselect_circle(stdimg);
    %masks=ROIselect_circle(ccimg);
    masks=ROIselect_circle(mat2gray(avgY));
    D = struct;
    D.(slice_no) = struct();
    D.(slice_no).(file_no) = struct();

    D.(slice_no).(file_no).masks = masks;
    D.(slice_no).(file_no).avgY = mat2gray(avgY);
    %D.(slice_no).(file_no).ccimg = ccimg;
    D.(slice_no).(file_no).curr_tiff = curr_tiff;
    D.(slice_no).(file_no).slice_path = tiff_dir;
    save(struct_fn, 'D')
    
else
    source_slice_no = 6; %8;
    source_file_no = 10; %2;
    %source_struct_fn = strcat(source_dir, sprintf('slice%i_traces_CC.mat', source_slice_no));
    source_struct_fn = strcat(source_dir, sprintf('slice%i_traces_AVG.mat', source_slice_no));
    source_struct = load(source_struct_fn);
    masks = source_struct.D.(['slice' num2str(source_slice_no)]).(['file' num2str(source_file_no)]).masks;
    %clear D;
end


%%
make_new_rois = 0;
nruns = 12; %12;
%slice_idx = source_slice_no;
source_file_no = 10;

slice_idx = 6;
run_idx = 10;


tiff_dir = strcat(source_dir, 'ch1_slice6/');
%%
source_file_no = 1;

% Choose .mat struct fn:
for fidx=1:length(tiffs)
    if fidx == source_file_no
        fprintf('skipping source...\n');
        continue
    end
    fprintf('Current run: %i\n', fidx)

% curr_tiff = sprintf('fov1_gratings_10reps_run1_Slice%02d_Channel01_File%03d.tif', slice_idx, run_idx);
% curr_tiff_path = strcat(tiff_dir, curr_tiff);
%curr_tiff = sprintf('fov1_gratings_%05d.tif #2.tif #%i.tif', run_idx, slice_idx);
%curr_tiff_path = strcat(tiff_dir, curr_tiff);


    curr_tiff = sprintf('fov2_rsvp_25reps_run3_Slice%i_Channel01_File%03d.tif', slice_idx, fidx);
    curr_tiff_path = fullfile(source_dir, tiff_dir, curr_tiff);

%avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
%ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
%stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);

slice_no = ['slice' num2str(slice_idx)];
file_no = ['file' num2str(fidx)]; % 'file1R'

% get average:
% avgimg = imread(strcat(source_dir, avgimg_fn));
% avgimg = mat2gray(avgimg);
% imshow(avgimg)

% get CC img to find active ROIs:
%ccimg = imread(strcat(source_dir, ccimg_fn));
%ccimg = mat2gray(ccimg);
% imshow(ccimg)

% get STD:
% stdimg = imread(strcat(source_dir, stdimg_fn));
% stdimg = mat2gray(stdimg);
% imshow(avgimg)

% % choose ROIs:
% if make_new_rois = 1
%     % masks=ROIselect_circle(stdimg);
%     masks=ROIselect_circle(ccimg);
%     % masks=ROIselect_circle(avgimg);
% else
%     source_slice_no = 8;
%     source_file_no = 2;
%     source_struct_fn = strcat(source_dir, sprintf('slice%i_traces_CC.mat', slice_no));
%     source_struct = load(source_struct_fn);
%     masks = source_struct.D.(source_slice_no).(source_file_no).masks;
%     %clear D;
% end

if ~exist(struct_fn)
    D = struct;
    D.(slice_no) = struct();
    D.(slice_no).(file_no) = struct();

    D.(slice_no).(file_no).masks = masks;
    %D.(slice_no).(file_no).avgimg = avgimg;
    %D.(slice_no).(file_no).ccimg = ccimg;
    D.(slice_no).(file_no).curr_tiff = curr_tiff_path;
    %D.(slice_no).(file_no).slice_path = tiff_dir;
    
    D.(slice_no).(file_no).source_file_no = source_file_no;
else
    load(struct_fn);
    if ~isfield(D.(slice_no), file_no)
        display('Current slice exists, but not current file.')
        D.(slice_no).(file_no) = struct();
    end
end


%
% load movie:
sframe=1;
Y = bigread2(curr_tiff_path,sframe);
if ~isa(Y,'double');    Y = double(Y);  end

%

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

% figure()
% for rtrace=1:size(raw_traces,1)
%     plot(raw_traces(rtrace,:), 'color', rand(1,3))
%     hold on;
% end

% if isfield(D.(slice_no).(file_no), 'raw_traces')
%     display(sprintf('Traces for file %s already exists.', file_no))
%     continue;
% else
    D.(slice_no).(file_no).raw_traces = raw_traces;
    save(struct_fn, 'D')
% end


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
if make_new_rois == 1
    roll_winsize = 25;
    %winsize = 25;
else
    roll_winsize = source_struct.D.(['slice' num2str(source_slice_no)]).(['file' num2str(source_file_no)]).roll_winsize; %25; %104; % 2 trials
    %winsize = source_struct.D.(['slice' num2str(source_slice_no)]).(['file' num2str(source_file_no)]).winsize;
end
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
%winsize = 25; %200; %185;

deltaF = zeros(size(raw_traces));
traces = zeros(size(raw_traces));
for rtrace=1:size(raw_traces,1)
    
    curr_trace = raw_traces(rtrace,:);
    %s1 = smooth(curr_trace, winsize, 'rlowess');
    %t1 = curr_trace - s1';

    %baseline = mean(t1);
    %dF = (t1 - baseline)./baseline; 
    
    baseline = mean(curr_trace);
    dF = (curr_trace - baseline) / baseline;
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
    %traces(rtrace,:) = t1;
end

%D.(slice_no).(file_no).winsize = winsize;
%D.(slice_no).(file_no).traces = traces;
D.(slice_no).(file_no).rolltraces = rolltraces;
D.(slice_no).(file_no).roll_winsize = roll_winsize;
D.(slice_no).(file_no).deltaF = deltaF;

save(struct_fn, 'D')


%% Find active cells:

%
% Choose cutoff by max peak:
peak_threshold = 0.005; %0.6 %0.05;
%peak_cutoff = 50 %100; %30; %200;
active_cells = [];
figure()
for rtrace=1:size(raw_traces,1)
    %check_max = (traces(rtrace,:) - mean(traces(rtrace,:)) ) ./ mean(traces(rtrace,:));
    %check_max = (traces(rtrace,:) - mean(traces(rtrace,:)) ) ./ mean(traces(rtrace,:));
    %if max(check_max) > peak_threshold
    %if (peak_cutoff > max(check_max) > peak_threshold)
    %if (peak_cutoff > max(deltaF(rtrace,:)) > peak_threshold)
    tmp_trace = rolltraces(rtrace,:);
    tmp_dF = (tmp_trace - mean(tmp_trace)) / mean(tmp_trace);
    
    if (max(deltaF(rtrace,:)) > peak_threshold)
    %if (max(traces(rtrace,:)) > peak_threshold)
        active_cells = [active_cells rtrace];
        plot(deltaF(rtrace,:)*100, 'color', rand(1,3))
        hold on;
    end
end
title(sprintf('%i ROIs found above threshold %i%%', length(active_cells), (peak_threshold*100)))

%

% Choose cutoff by max peak:
peak_threshold = 0.005; %0.6 %0.05;
%peak_cutoff = 50 %100; %30; %200;
active_cells = [];
figure()
for rtrace=1:size(raw_traces,1)
    %check_max = (traces(rtrace,:) - mean(traces(rtrace,:)) ) ./ mean(traces(rtrace,:));
    %check_max = (traces(rtrace,:) - mean(traces(rtrace,:)) ) ./ mean(traces(rtrace,:));
    %if max(check_max) > peak_threshold
    %if (peak_cutoff > max(check_max) > peak_threshold)
    %if (peak_cutoff > max(deltaF(rtrace,:)) > peak_threshold)
    if (max(deltaF(rtrace,:)) > peak_threshold)
    %if (max(traces(rtrace,:)) > peak_threshold)
        active_cells = [active_cells rtrace];
        plot(deltaF(rtrace,:)*100, 'color', rand(1,3))
        hold on;
    end
end
title(sprintf('%i ROIs found above threshold %0.2f%%', length(active_cells), (peak_threshold*100)))

D.(slice_no).(file_no).active_cells = active_cells;
D.(slice_no).(file_no).peak_threshold = peak_threshold;
%D.(slice_no).(file_no).peak_cutoff = peak_cutoff;
save(struct_fn, 'D')

end

%%



%% Match traces to stimuli:

% Get MW and ARD info from python .mat file:
% pymat_fn = 'fov1_gratings_10reps_run1.mat';
% 
% pymat_fn = '20161219_JR030W_gratings_bluemask_5trials_2.mat';
% S = load(strcat(source_dir, 'mw_data/', pymat_fn));

% pymat_fn = '20161222_JR030W_gratings_10reps_run1.mat';
% S = load(strcat(source_dir, 'mw_data/', pymat_fn));
pymat_fn = '20160118_AG33_gratings_fov1_run10.mat';
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


first_struct_slice = 'slice6'; %'slice7';
first_struct_file = 'file10'; %'file2';
%first_struct = load(strcat(source_dir, 'slice7_traces_cc2.mat'));
first_struct = load(strcat(source_dir, 'slice6_traces_AVG.mat'));
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
% mw_start_tif = 2; % if acquis. file incrementer not re-started (i.e,. threw out _00001.tif)
% ndiscard = 5; % removed previous _0000x.tifs due to scan phase adj (12/21-12/22)
% mw_fidx = 1%fidx + ndiscard
%    
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

% run_idx = 2;
% ntrash = 1;
% start_tif = 7;
% tif_no = start_tif - ntrash;
% ndiscard = 5;
% mc_file_no = 1;
% mw_fidx = 6 + run_idx - 1; % tif_no - ntrash; %mc_file_no+ntrash;

separate_runs = 0;


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
for pyfn=1:length(run_fns)
    if strfind(run_fns(pyfn).name, curr_run_suffix)
        pymatfn = run_fns(pyfn).name;
    end
end
mw_fidx = 1;

end
S = load(strcat(source_dir, 'mw_data/', pymat_fn));


%pymat_fn = sprintf('20160115_AG33_gratings_fov3_run%i.mat', ridx);
%pymat_fn = sprintf('20160118_AG33_gratings_fov1_run%i.mat', run_idx);
%S = load(strcat(source_dir, pymat_fn));

ard_filedurs = double(S.ard_file_durs);
mw_times = S.mw_times_by_file;
offsets = double(S.offsets_by_file);
mw_codes = S.mw_codes_by_file;
mw_filedurs = double(S.mw_file_durs);



mw_fidx = 1; % AG33 data

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
    y_sec = interp1([0:length(ard_sec)-1], ard_sec, linspace(0, length(ard_sec)-1, nframes));
end
    
    

% save, if needed:
slice_no = 8;
file_no = 2;

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
curr_slice = 6%8; %13; %19; %11;
curr_run = 10 %2; %3; %6; %10; %1;
no_mw = 0;

%D_traces = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).deltaF;
%D_traces = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).traces;
D_traces = D.(sprintf('slice%i', curr_slice)).(sprintf('file%i', curr_run)).deltaF;
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
   cell = 3; %pidx; %pidx % + 30;
   
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



%% Plot ROIs and show active cells:

figure()
% nparts = strsplit(tseries_fn, '_');
% figname = strcat(nparts(1), nparts(2), nparts(3), nparts(4), nparts(5), nparts(7));

nparts = strsplit(curr_tiff, ' ');
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
