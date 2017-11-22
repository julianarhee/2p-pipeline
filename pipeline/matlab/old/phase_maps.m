


%% Weird oscillations?

%cell = 7 %(fov2_bar5_00002)
%S = D_traces(D_active(cell), :);

S = rawslice;% raw_traces(6,:); %rolled;%vals;%y;  %t1; %traces(6,:);

Fs = 5.58;
T = 1/Fs;d
L = length(S);
t = (0:L-1)*T;

% figure()
% plot(t, S, 'k', 'LineWidth', 1)

ft = fft(S);
P2 = abs(ft/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);


figure()
target_freq = 0.37;
f1 = Fs*(0:(L/2))/L;
plot(f1(2:end),P1(2:end))
hold on;
freq_idx = find(abs(f1-target_freq) == min(abs(f1-target_freq)));
plot(f1(freq_idx), P1(freq_idx), 'r*')
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%%

%% Define Source:

run_idx = 5;
cond_idx = 3;
source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/fov2_bar5/';


run_idx = 1;
cond_idx = 1;
source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20170118_AG33/fov1_bar008Hz_test1/';


run_idx = 1;
cond_idx = 1;
source_dir = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopy/fov6_retinobar_037Hz_final_nomask/';


% --------
run_idx = 4;
cond_idx = 3; %4;
nbad = 2; % n of MW runs to ignore due to bad scan phase adjustment runs
mw_idx = cond_idx + nbad;
source_dir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/retinotopy037Hz/fov1_bar037Hz_run4/';

% -----

% -------------------------------------------------------------------------
% Set here only for source files to be used for getting masks:
% nslices = 20;
nslices = 20; %20; %30; %22
ndiscard = 8; %0; %8;
ntotal_slices = nslices + ndiscard;

nvolumes = 455;
%nvolumes = 340; %340; %350; %340; %1952;
nframes = nvolumes * ntotal_slices;
nchannels = 1;
ntotal_frames = nframes * nchannels;

%% Read in TIFF volumes:
%

tiff_names = dir(fullfile(source_dir, '*.tif'));
tiff_names = {tiff_names(:).name};

% -------------------------------------------------------------------------
% 1.  Split channels (if no motion correction, raw data have Ch1/Ch2
% interleaved):

two_channel = 1;

sframe = 1;
for tiff_idx = 1:length(tiff_names)
    curr_tiff_path = fullfile(source_dir, tiff_names{tiff_idx});
    Y = bigread2(curr_tiff_path,sframe, ntotal_frames);
    
    if two_channel==1
        Y_ch1 = Y(:,:,1:2:end); %This is doing sth weird, ch1 becomes ch2, and images look shittier.
        Y_ch2 = Y(:,:,2:2:end);

        parsed_tiff_fn1 = [tiff_names{tiff_idx} ' #1.tif'];
        parsed_tiff_fn2 = [tiff_names{tiff_idx} ' #2.tif'];
        
        ch1_dir = fullfile(source_dir, 'Channel01'); 
        ch2_dir = fullfile(source_dir, 'Channel02');
        if ~exist(ch1_dir, 'dir')
            mkdir(ch1_dir);
            mkdir(ch2_dir);
        end
        try
            tiffWrite(Y_ch1, parsed_tiff_fn1, ch1_dir);
            tiffWrite(Y_ch2, parsed_tiff_fn2, ch2_dir);
        catch
            pause(60);
            tiffWrite(Y_ch1, parsed_tiff_fn1, ch1_dir);
            tiffWrite(Y_ch2, parsed_tiff_fn2, ch2_dir);
        end
    end
end
            
%% Get MW info:

conds = {'bottom', 'left', 'right', 'top'};
% conds = {'right'};
%conds = {'right', 'top', 'bottom', 'left'};
%conds = {'top', 'bottom', 'right', 'top', 'bottom', 'left'};
curr_cond_no = cond_idx;

pymat_fn = '20161218_CE025_bar5.mat';
%pymat_fn = '20170118_AG33_bar008Hz_fov1_test1.mat';
%pymat_fn = '20161219_JR030W_retinobar_037Hz_final_fov6_nomask.mat';
%pymat_fn = '20161221_JR030W_bar037Hz_run4.mat';

S = load(strcat(source_dir, 'mw_data/', pymat_fn));
curr_cond = conds{curr_cond_no};

mw_times = double(S.(curr_cond).time);
mw_sec = (mw_times - mw_times(1)) / 1000000;
cycle_starts = S.(curr_cond).idxs + 1;
dur = (double(S.triggers(curr_cond_no,2)) - double(S.triggers(curr_cond_no,1))) / 1000000;

cut_end = 1;
ncycles = 30;
target_freq = 0.37
Fs = 5.58%4.11 %4.26 %5.58
crop = ceil((1/target_freq)*ncycles*Fs);

% y_sec = linspace(0, dur, ntotal_frames);
y_sec_vols = linspace(0, dur, ntotal_frames);
if cut_end==1
    y_sec = linspace(0, dur, crop);
else
    y_sec = linspace(0, dur, nvolumes);
end
last = ceil(find(abs(y_sec - mw_sec(end)) == min(abs(y_sec - mw_sec(end)))));

%crop = last+1;

%% Get TIFF stack:

check_slice = 0;  %
%run_idx = 4; %1; %5;
slice_idx = 15; %10; %10;
create_rois = 1;
use_previous_masks=0;

if check_slice==1
    slice_no = sprintf('slice%i', slice_idx);
    %curr_tiff_fn = sprintf('fov2_bar5_%05d.tif #1.tif #%i.tif', cond_idx, slice_idx);
% 
%     curr_tiff_fn = sprintf('fov1_bar_%05d.tif #%i.tif', cond_idx, slice_idx);
%     curr_tiff_fn = fullfile('ch2_slices', curr_tiff_fn);
%   
    curr_tiff_fn = sprintf('fov1_bar037Hz_run%i_Slice%i_Channel01_File%03d.tif', run_idx, slice_idx, cond_idx);
    %curr_tiff_fn = sprintf('fov1_bar037Hz_run%i_Slice%i_Channel01_File%03d_uint16.tif', run_idx, slice_idx, cond_idx);

    if create_rois==1
        acquisition_struct_fn = sprintf('bar_cond%i_sl%i_ROIs.mat', cond_idx, slice_idx)
    else
        if use_previous_masks==1
            orig_struct_fn = 'fov1_bar037Hz_run4/bar_cond3_sl10_100ROIs_uint16.mat';
            orig_struct = load(fullfile('/nas/volume1/2photon/RESDATA/20161221_JR030W/retinotopy037Hz/', orig_struct_fn));
            masks = orig_struct.acquisition.masks;        
            acquisition_struct_fn = sprintf('bar_cond%i_sl%i_100ROIs_uint16_test.mat', cond_idx, slice_idx)

        else
            acquisition_struct_fn = sprintf('bar_cond%i_sl%i_pix.mat', cond_idx, slice_idx)
            masks = 'pixels';
        end
    end
else

    curr_tiff_fn = sprintf('fov2_bar5_%05d.tif #1.tif', cond_idx);
    % curr_tiff_fn = sprintf('fov1_bar_%05d.tif', cond_idx);
    %curr_tiff_fn = sprintf('fov6_retinobar_037Hz_final_nomask_%05d.tif #1.tif', cond_idx);
    
    
    if create_rois==1
        acquisition_struct_fn = sprintf('bar_cond%i_sl%i_100ROIs_test.mat', cond_idx, slice_idx)
    else
        acquisition_struct_fn = sprintf('bar_cond%i_slice%i_test_pix.mat', cond_idx, slice_idx)
        masks = 'pixels';
    end
    
end
curr_tiff_path = fullfile(source_dir, curr_tiff_fn);


cond_no = sprintf('cond%i', cond_idx);
run_no = sprintf('run%i', run_idx);

% -------------------------------------------------------------------------
% load tiff
sframe = 1;
Y = bigread2(curr_tiff_path,sframe);
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;      


%[YY,~] = tiffRead(curr_tiff_path);

%acquisition_struct_fn = sprintf('bar%i_cond%i_FFT_k3_slice%i.mat', run_idx, cond_idx, slice_idx)
%acquisition_struct_fn = sprintf('bar_cond%i.mat', cond_idx)

% 
% [cc]=CrossCorrImage(Y);
% cc_fn = strcat(source_dir, 'CC/cc_', curr_tiff_fn(1:end-4), '.png');
% imwrite(cc, cc_fn);

if check_slice == 1
    avgY = mean(Y,3);
else
    avgY = mean(Y(:,:,slice_idx:ntotal_slices:ntotal_frames),3);
end
imagesc(avgY);
colorbar()
%% Resample binned pixels:

smooth_pixels = 0;
use_previous_masks = 0; %1;
create_rois = 1;
    
if create_rois == 0 && use_previous_masks==0

    ksize = 4;

    if smooth_pixels==1
        binY = zeros(size(Y));
        for frame=1:size(Y,3);
            curr_frame = Y(:,:,frame);
            padY = padarray(curr_frame, [ksize, ksize], 'replicate');
            convY = conv2(padY, fspecial('average',[ksize ksize]), 'same');
            binY(:,:,frame) = convY(ksize+1:end-ksize, ksize+1:end-ksize);
        end
        raw_traces = binY;
    else
        raw_traces = Y;
    end
end

%% Create new ROIs:

use_previous_masks = 0; %1;
create_rois = 1;
    
if create_rois == 1

%     %avgimg_fn = 'AVG_fov1_bar037Hz_run4_Slice10_Channel01_File003.tif';
%     avgimg_fn = 'STD_fov1_bar037Hz_run4_Slice10_Channel01_File003_uint16.tif';
%     avgimg = imread(fullfile(source_dir, 'AVG', avgimg_fn));
%     
%     avgimg = mat2gray(avgimg);
    if use_previous_masks==0
        masks=ROIselect_circle(mat2gray(avgY));
    end
    
end

if smooth_pixels == 0 
% extract raw traces:
%raw_traces = zeros(size(masks,3), size(Y,3));
raw_traces = zeros(size(masks,3), size(Y,3));
for r=1:size(masks,3)
    curr_mask = masks(:,:,r);
    Y_masked = nan(1,size(Y,3));
    for t=1:size(Y,3)
        t_masked = curr_mask.*Y(:,:,t);
        t_masked(t_masked==0) = NaN;
        Y_masked(t) = nanmean(t_masked(:));
        %Y_masked(t) = sum(t_masked(:));
    end
    raw_traces(r,:,:) = Y_masked;
end
end

% acquisition_struct_fn = strcat(acquisition_struct_fn(1:end-4), '_medROIs.mat')
%%
acquisition = struct(); % save acquisition struct
acquisition.masks = masks;
acquisition.avgimg = avgY;
%acquisition.avgimg_fn = avgimg_fn;


acquisition.trim_series = cut_end;
acquisition.true_nsamples = crop;

%acquisition.run = run_no;
acquisition.condition = cond_idx;
acquisition.condname = conds{cond_idx};
acquisition.tiff_path = curr_tiff_path;
acquisition.tiff_size = size(Y);
acquisition.smoothed = smooth_pixels;

if smooth_pixels
    acquisition.kernel = ksize;
else
    acquisition.kernel = 'none';
end
acquisition.volume = raw_traces;
save(fullfile(source_dir, acquisition_struct_fn), 'acquisition');



acquisition.pymat_path = strcat(source_dir, 'mw_data/', pymat_fn);
acquisition.pymat = S;
acquisition.y_sec = y_sec;
acquisition.y_sec_vols = y_sec_vols;
acquisition.mw_sec = mw_sec;
acquisition.mw_cycle_idxs = cycle_starts;
save(fullfile(source_dir, acquisition_struct_fn), 'acquisition');


% Save ROI nums to img:
% nparts = strsplit(curr_tiff_fn, ' ');
% %figname = strcat(nparts(1), '_', slice_no, '_', run_no);
% figname = strcat(nparts(1), '_', 'VOL', '_', run_no);
%%
figname = strcat('ROIs_', acquisition_struct_fn(1:end-4), '.png');
figname = strrep(figname, '_', '-')

RGBimg = zeros([size(avgY),3]);
RGBimg(:,:,1)=0;
RGBimg(:,:,2)=mat2gray(avgY); %mat2gray(avgY);
RGBimg(:,:,3)=0;

%figure()
numcells=size(masks,3);
for c=1:numcells
    RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,c);
end

fig = figure();
imshow(RGBimg);
title(figname);
hold on;
for c=1:numcells
    [x,y] = find(masks(:,:,c)==1,1);
    text(y,x,num2str(c))
    hold on;
end

figdir = fullfile(source_dir, 'figures', acquisition_struct_fn(1:end-4));
if ~exist(figdir, 'dir')
    mkdir(figdir)
end

saveas(fig, fullfile(figdir, figname));

saveas(fig, fullfile(figdir, strcat(figname(1:end-4),'.fig')));


acquisition.RGBimg = RGBimg;
save(acquisition_struct_fn, 'acquisition')


% 
% A = rand(120,120);
% convA = conv2(A, ones(3,3), 'same');
% figure(); subplot(1,2,1); imagesc(A); hold on; subplot(1,2,2); imagesc(convA)
% 
% padA = padarray(A, [3, 3], 'replicate');
% convA = conv2(padA, fspecial('average',[3 3]), 'same');
% figure(); subplot(1,2,1);
% imagesc(A);
% colorbar()
% hold on;
% subplot(1,2,2);
% imagesc(convA(3:end-3, 3:end-3));
% colorbar()

%%  PHASE MAP -- ROIs or ALL PIXELS:

%crop = 336;

use_rois = 1;
check_slice=0;
    
target_freq = 0.37;
Fs = 5.58; % 4.11; %4.26; %5.58; % 4.11; %4.26; %5.58;

cut_end=1
crop=round((1/0.37)*30*Fs); %+2

winsz = round((1/target_freq)*Fs*2);

phase_map = zeros(d1, d2, 1);
mag_map = zeros(d1, d2, 1);
max_map = zeros(d1, d2, 1);


if use_rois==0
    
fft_struct = struct();
for row=1:d1;
    fprintf('Processing Row #: %i\n', row);
    for col=1:d2;
        roi_no = sprintf('roi_x%i_y%i', row, col);
    
        vol_trace = raw_traces(row, col, :);
        for slice=slice_idx:slice_idx %ntotal_slices;
            slice_indices = slice:ntotal_slices:ntotal_frames;
            vol_offsets = y_sec_vols(slice_indices);

            tmp0 = zeros(1,length(slice_indices));
            if check_slice==1
                tmp0(:) = squeeze(vol_trace(1:end)); % don't use volume slice indices if just loading in 1 slice
            else
                tmp0(:) = squeeze(vol_trace(:,:,slice_indices));
            end
            if cut_end==1
                tmp0 = tmp0(1:crop);
            end
            tmp1 = padarray(tmp0,[0 winsz],tmp0(1),'pre');
            tmp1 = padarray(tmp1,[0 winsz],tmp1(end),'post');
            rollingAvg=conv(tmp1,fspecial('average',[1 winsz]),'same');%average
            rollingAvg=rollingAvg(winsz+1:end-winsz);
            trace_y = tmp0 - rollingAvg;
        
            NFFT = length(trace_y);
            fft_y = fft(trace_y,NFFT);
            freqs = Fs*(0:(NFFT/2))/NFFT;
            freq_idx = find(abs((freqs-target_freq))==min(abs(freqs-target_freq)));

            magY = abs(fft_y);
            %phaseY = unwrap(angle(Y));
            phaseY = angle(fft_y);
            %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
            %phase_deg = phase_rad / pi * 180 + 90;

            fft_struct.(roi_no).targetPhase = phaseY(freq_idx); % unwrap(angle(Y(freq_idx)));
            fft_struct.(roi_no).targetMag = magY(freq_idx);

            fft_struct.(roi_no).fft_y = fft_y;
            fft_struct.(roi_no).DC_y = trace_y;
            fft_struct.(roi_no).raw_y = tmp0;
            fft_struct.(roi_no).slices = slice_indices;
            fft_struct.(roi_no).freqs = freqs;
            fft_struct.(roi_no).freq_idx = freq_idx;
            fft_struct.(roi_no).target_freq = target_freq;
            
            phase_map(row, col) = phaseY(freq_idx);
            mag_map(row, col) = magY(freq_idx);
            max_idx = find(magY==max(magY));
            max_map(row, col) = phaseY(max_idx(1));
        end
    end
    
end

else
% PHASE MAP -- ROIs
% 
% raw_traces = acquisition.volume;
% y_sec_vols = acquisition.y_sec_vols;
% y_sec = acquisition.y_sec;
% cycle_starts = acquisition.mw_cycle_idxs;
% mw_sec = acquisition.mw_sec;
% masks = acquisition.masks;
% 
% phase_map = zeros(d1, d2, 1);
% mag_map = zeros(d1, d2, 1);
% max_map = zeros(d1, d2, 1);


fft_struct = struct();
for row=1:size(raw_traces,1)
    fprintf('Processing Row #: %i\n', row);
        roi_no = sprintf('roi_%i', row);
        
        vol_trace = raw_traces(row, :);
        
        for slice=slice_idx:slice_idx %ntotal_slices;
            slice_indices = slice:ntotal_slices:ntotal_frames;
            vol_offsets = y_sec_vols(slice_indices);

            tmp0 = zeros(1,length(slice_indices));
            if check_slice==1
                tmp0(:) = squeeze(vol_trace(1:end)); % don't use volume slice indices if just loading in 1 slice
            else
                tmp0(:) = squeeze(vol_trace(slice_indices));
            end
            if cut_end==1
                tmp0 = tmp0(1:crop);
            end
            tmp1 = padarray(tmp0,[0 winsz],tmp0(1),'pre');
            tmp1 = padarray(tmp1,[0 winsz],tmp1(end),'post');
            rollingAvg=conv(tmp1,fspecial('average',[1 winsz]),'same');%average
            rollingAvg=rollingAvg(winsz+1:end-winsz);
            trace_y = tmp0 - rollingAvg;
        
            NFFT = length(trace_y);
            fft_y = fft(trace_y,NFFT);
            %F = ((0:1/NFFT:1-1/NFFT)*Fs).';
            freqs = Fs*(0:(NFFT/2))/NFFT;
            freq_idx = find(abs((freqs-target_freq))==min(abs(freqs-target_freq)));

            magY = abs(fft_y);
            %phaseY = unwrap(angle(Y));
            phaseY = angle(fft_y);
            %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
            %phase_deg = phase_rad / pi * 180 + 90;

            fft_struct.(roi_no).targetPhase = phaseY(freq_idx); % unwrap(angle(Y(freq_idx)));
            fft_struct.(roi_no).targetMag = magY(freq_idx);

            fft_struct.(roi_no).fft_y = fft_y;
            fft_struct.(roi_no).DC_y = trace_y;
            fft_struct.(roi_no).raw_y = tmp0;
            fft_struct.(roi_no).slices = slice_indices;
            fft_struct.(roi_no).freqs = freqs;
            fft_struct.(roi_no).freq_idx = freq_idx;
            fft_struct.(roi_no).target_freq = target_freq;
            
            phase_map(masks(:,:,row)==1) = phaseY(freq_idx);
            mag_map(masks(:,:,row)==1) = magY(freq_idx);
            max_idx = find(magY==max(magY));
            max_map(masks(:,:,row)==1) = phaseY(max_idx(1));
        end
    
end

end
%%

if smooth_pixels==0
    binY = Y;
end
if check_slice == 1
    if exist('binY')
    curr_chunk = binY;
    avgY = mean(binY,3);
    else
    curr_chunk = Y;
    avgY = mean(Y, 3);
    end
else
    curr_chunk = binY(:,:,slice_indices);
    avgY = mean(binY(:,:,slice_indices),3);
end
maxY = max(curr_chunk(:));
minY = min(curr_chunk(:));

x_scaled = (avgY-min(avgY(:)))*(maxY-minY)/(max(avgY(:))-min(avgY(:))) + minY;

FFT = struct();
FFT.sampling_rate = Fs;
FFT.DC_window = winsz;
FFT.avgimg = avgY;
FFT.fft_struct = fft_struct;

FFT.phase_map = phase_map;
FFT.mag_map = mag_map;
FFT.max_map = mag_map;
FFT.max_idx = max_idx;

slice_acquisition_struct_fn = strcat('FFT_', acquisition_struct_fn(1:end-4), sprintf('_slice%i', slice_idx), '.mat');
save(fullfile(source_dir, slice_acquisition_struct_fn), 'FFT', '-v7.3');


tiffWrite(curr_chunk, sprintf('binned_sliced_slice%i', slice_idx), source_dir);
%
%% fake legend:

sim_acquisition = zeros(10,50);
sim_acquisition(:,1) = ones(size(sim_acquisition(:,1)));
legend_im = zeros(10,50,100);
legend_im(:,:,1) = sim_acquisition(:,:,1);
tmpA = sim_acquisition;
for lidx=2:100
    legend_im(:,:,lidx) = circshift(tmpA, 1, 2);
    tmpA = legend_im(:,:,lidx);
end

Fs_lgd = 1;
T_lgd = 1/Fs_lgd;
L_lgd = size(legend_im,3);
t_lgd = (0:L_lgd-1)*T_lgd;
target_freq_lgd = 1/50;
legend_phase = zeros(size(legend_im,1), size(legend_im,2));
for r_lgd=1:size(legend_im,1)
    for c_lgd=1:size(legend_im,2)
        y_lgd = legend_im(r_lgd,c_lgd,:);
        NFFT_lgd = length(y_lgd);
        legend_ft = fft(y_lgd,NFFT_lgd);
        %freqs = Fs*(0:1/(NFFT/2))/NFFT;
        freqs_lgd = ((0:1/NFFT_lgd:1-1/NFFT_lgd)*Fs_lgd).';
        freq_idx_lgd = find(abs((freqs_lgd-target_freq_lgd))==min(abs(freqs_lgd-target_freq_lgd)));
        magY = abs(legend_ft);
        legend_phase(r_lgd,c_lgd) = angle(legend_ft(freq_idx_lgd)); % unwrap(angle(Y(freq_idx)));
    end        
end

figure()
imagesc(legend_phase)
colormap hsv
caxis([-pi, pi])
%colorbar()
axis('off')

legends = struct();
legends.left = legend_phase;
legends.right = fliplr(legend_phase);




% HORIZONTAL:
sim_acquisition = zeros(10,50);
sim_acquisition(1,:) = ones(size(sim_acquisition(1,:)));
legend_im = zeros(10,50,100);
legend_im(:,:,1) = sim_acquisition(:,:,1);
tmpA = sim_acquisition;
for lidx=2:100
    legend_im(:,:,lidx) = circshift(tmpA, 1,1);
    tmpA = legend_im(:,:,lidx);
end

Fs_lgd = 1;
T_lgd = 1/Fs_lgd;
L_lgd = size(legend_im,3);
t_lgd = (0:L_lgd-1)*T_lgd;
target_freq_lgd = 1/10;
legend_phase = zeros(size(legend_im,1), size(legend_im,2));
for r_lgd=1:size(legend_im,1)
    for c_lgd=1:size(legend_im,2)
        y_lgd = legend_im(r_lgd,c_lgd,:);
        NFFT_lgd = length(y_lgd);
        legend_ft = fft(y_lgd,NFFT_lgd);
        %freqs = Fs*(0:1/(NFFT/2))/NFFT;
        freqs_lgd = ((0:1/NFFT_lgd:1-1/NFFT_lgd)*Fs_lgd).';
        freq_idx_lgd = find(abs((freqs_lgd-target_freq_lgd))==min(abs(freqs_lgd-target_freq_lgd)));
        magY = abs(legend_ft);
        legend_phase(r_lgd,c_lgd) = angle(legend_ft(freq_idx_lgd)); % unwrap(angle(Y(freq_idx)));
    end        
end

figure()
imagesc(legend_phase)
colormap hsv
caxis([-pi, pi])
%colorbar()
axis('off')

legends.top = legend_phase;
legends.bottom = flipud(legend_phase);

FFT.legends = legends;
save(fullfile(source_dir, slice_acquisition_struct_fn), 'FFT', '-v7.3');


%%
load_datastruct = 0;
if load_datastruct==1
    dstruct = 'bar_cond3_sl10_ROIs_uint16.mat';
    load(fullfile(source_dir, dstruct));
end

% % y_sec = linspace(0, dur, ntotal_frames);
% if cut_end==1
%     y_sec = linspace(0, dur-1, crop);
% else
%     y_sec = linspace(0, dur-1, nvolumes);
% end
% % last = ceil(find(abs(y_sec - mw_sec(end)) == min(abs(y_sec - mw_sec(end)))));
% % 
% % crop = last+1;




fig = figure();
%A = repmat(x_scaled, [1, 1, 3]);
ax1 = subplot(2,2,1);
imagesc(x_scaled);
axis('off')
hb = colorbar('location','eastoutside');
colormap(ax1, gray)

ax4 = subplot(2,2,4);
threshold = 0.05; %8000; %10000; %8000; %(k=3); %20000;
threshold_map = phase_map;
threshold_map(mag_map<(max(mag_map(:))*threshold)) = NaN;

fov = repmat(mat2gray(avgY), [1, 1, 3]);
B = threshold_map; %phase_map
imagesc(fov);
slice_no = 'slice4';
title(sprintf('avg - %s', slice_no))
hold on;
Bimg = imagesc2(B);
title('phase')
colormap(ax4, hsv)
caxis([-pi, pi])


ax3 = subplot(2,2,3);
imagesc(mag_map)
axis('off')
colormap(ax3, hot)
hb = colorbar('location','eastoutside');
title('magnitude')

ax2 = subplot(2,2,2);
imagesc(legends.(conds{cond_idx}))
axis('off')
caxis([-pi, pi])
colormap(ax2, hsv)

% 
% mainPos = get(ax4,'position');
% 
% ax1Pos = get(ax1,'position');
% ax1Pos(3:4) = [mainPos(3:4)];
% set(ax1,'position',ax1Pos);
% 
% ax3Pos = get(ax3,'position');
% ax3Pos(3:4) = [mainPos(3:4)];
% set(ax3,'position',ax3Pos);
% 
% ax2Pos = get(ax2,'position');
% ax2Pos(3:4) = [mainPos(3:4)];
% set(ax2,'position',ax2Pos);

figdir = fullfile(source_dir, 'figures', acquisition_struct_fn(1:end-4));
if ~exist(figdir, 'dir')
    mkdir(figdir)
end


figname = strcat('FFT_', acquisition_struct_fn(1:end-4))
saveas(fig, fullfile(figdir, figname));
figname = strcat('FFT_', acquisition_struct_fn(1:end-4), '.png')
saveas(fig, fullfile(figdir, figname));


% -------------------------------------------------------------------------
% Save just PHASE:

fig = figure();
hold on;
imagesc(fov);
slice_no = sprintf('slice%i', slice_idx);
%title(sprintf('avg - %s', slice_no))
hold on;
Bimg = imagesc2(B);
%title('phase')
colormap hsv
caxis([-pi, pi])

fig.PaperPositionMode = 'auto'
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

figname = strcat('phase_', acquisition_struct_fn(1:end-4), '.fig')
saveas(fig, fullfile(figdir, figname));
% export_fig([fullfile(figdir, figname),'.pdf'], '-pdf') %,'-transparent');
% print(fig,figname,'-dpdf')
figname = strcat('phase_', acquisition_struct_fn(1:end-4), '.bmp')
saveas(fig, fullfile(figdir, figname));


% LEGEND: ----------------
fig= figure();
imagesc(legends.(conds{cond_idx}))
axis('off')
caxis([-pi, pi])
colormap hsv

figname = strcat('phase_', acquisition_struct_fn(1:end-4), '_legend.bmp')
saveas(fig, fullfile(figdir, figname));


%% Check active rois:

%fft_struct = F.fft;
rois = fieldnames(fft_struct);

%vol_offset = vol_offsets(15);
% rawtrace = fft_struct.(rois{roi}).raw_y;
% dc_sub = fft_struct.(rois{roi}).DC_y;

min_df = 4; %20; %0.5;
active_rois = [];
active_maxima = [];
for roi=1:length(rois)
    rawtrace = fft_struct.(rois{roi}).raw_y;
    dc_sub = fft_struct.(rois{roi}).DC_y;
    shift_y = dc_sub + mean(rawtrace);
    dF = (shift_y - mean(shift_y)) ./ mean(shift_y);
    %display(max(dF(:)))
    if max(dF(:))*100 > min_df
        active_rois = [active_rois roi];
    end
    active_maxima = [active_maxima max(dF(:))];
end
fprintf('Found %i ROIs with dF/F > %0.2f%%.\n', length(active_rois), min_df);

amax = active_maxima(active_rois);
[amaxY, amaxI] = sort(amax, 'descend');
sorted_rois = active_rois(amaxI);

%%
% -------------------------------------------------------------------------
% Look at power spectra:

plot_semilog = 0
fig = figure();
rois = fieldnames(fft_struct);
pidx = 1;

% check_rois = active_rois(randi(9, [1,9])); %randi(length(rois), [1 9]);
check_rois = sorted_rois(1:9);

for ridx=1:length(check_rois); %1:length(rois)
    subplot(3,3, pidx)
    
    roi = check_rois(ridx);
    
    curr_fft = fft_struct.(rois{roi}).fft_y;
    L = length(curr_fft);
    mags = abs(curr_fft) / L;
    power = mags.^2;
    freqs = fft_struct.(rois{roi}).freqs;
    freq_idx = fft_struct.(rois{roi}).freq_idx;
    
    P1 = power(1:L/2+1);
    %P1(2:end-1) = 2*P1(2:end-1);
    
    if plot_semilog==1
        %semilogx(fax_Hz(1:N_2), 20*log10(X_mags(1:N_2)))
        semilogx(freqs, 20*log10(mags(1:L/2+1)), 'k')
        hold on
        %semilogx(freqs(freq_idx), 20*log10(mags(freq_idx)), 'r*')
        semilogx(freqs(freq_idx), max(20*log10(mags)), 'r*')
        ylabel('power (dB)')
        xlabel('F (Hz)')
        figname_prefix = 'power-dB';
    else
        plot(freqs, P1, 'k')
        hold on
        plot(freqs(freq_idx), max(P1), 'r*')
        %plot(freqs(freq_idx), P1(freq_idx), 'r*')
        ylabel('power')
        xlabel('F (Hz)')
        figname_prefix = 'power';
    end
    title(strrep(rois{roi}, '_', '-'))
    pidx = pidx + 1;
end

plotname = strrep(acquisition_struct_fn(1:end-4), '_', '-');
suptitle(plotname)

figname = strcat(figname_prefix, '_', strrep(acquisition_struct_fn(1:end-4), '_', '-'))
saveas(fig, fullfile(figdir, figname));


% -------------------------------------------------------------------------
% Look at PHASE map by ROIs:
phaseMap = zeros(size(avgY));
phaseMap(phaseMap==0) = NaN;
for roi=1:length(fieldnames(fft_struct))
    
    roi_no = sprintf('roi%i', roi);
    curr_phase = fft_struct.(roi_no).targetPhase;
    curr_mag = fft_struct.(roi_no).targetMag;
    replaceNan = masks(:,:,roi)==1;
    phaseMap(replaceNan) = 0;
    phaseMap = phaseMap + curr_phase*masks(:,:,roi);
    
end

fig = figure();
acquisition = repmat(avgY, [1, 1, 3]);
B = phaseMap;
subplot(1,2,1)
imagesc(acquisition);
hold on;
Bimg = imagesc2(B);
colormap hsv
caxis([-pi, pi])
%colorbar()

% legend:

figname = strcat(figname_prefix, '_', strrep(acquisition_struct_fn(1:end-4), '_', '-'))
saveas(fig, fullfile(figdir, figname));


%% PLot time series with stimulus onsets:

%roi = 10; %; 6;

% 
% fft_struct = F.fft;
% rois = fieldnames(fft_struct);
% 
% %vol_offset = vol_offsets(15);
% % rawtrace = fft_struct.(rois{roi}).raw_y;
% % dc_sub = fft_struct.(rois{roi}).DC_y;
% 
% min_df = 5; %0.5;
% active_rois = [];
% for roi=1:length(rois)
%     rawtrace = fft_struct.(rois{roi}).raw_y;
%     dc_sub = fft_struct.(rois{roi}).DC_y;
%     shift_y = dc_sub + mean(rawtrace);
%     dF = (shift_y - mean(shift_y)) ./ mean(shift_y);
%     %display(max(dF(:)))
%     if max(dF(:))*100 > min_df
%         active_rois = [active_rois roi];
%     end
% end
% fprintf('Found %i ROIs with dF/F > %0.2f%%.\n', length(active_rois), min_df);

%

for ridx=1:9
roi = sorted_rois(ridx); 
rawtrace = fft_struct.(rois{roi}).raw_y;
dc_sub = fft_struct.(rois{roi}).DC_y;


fig = figure(); 
subplot(3,1,1)
plot(y_sec(1:length(rawtrace)), rawtrace, 'k'); hold all;
%plot(mw_sec(cycle_starts)-vol_offsets(15), ones(size(mw_sec(cycle_starts)))*mean(rawslice), 'r.')
y1 = get(gca,'ylim');
for cyc=1:length(cycle_starts)
    x = mw_sec(cycle_starts(cyc));
    line([x x], [y1(1) y1(2)], 'Color', 'r');
end
title('raw')

subplot(3,1,2)
plot(y_sec(1:length(dc_sub)), dc_sub, 'k')
%plot(mw_sec(cycle_starts)-vol_offset, ones(size(mw_sec(cycle_starts)))*mean(y), 'r.');
y1 = get(gca,'ylim');
for cyc=1:length(cycle_starts)
    x = mw_sec(cycle_starts(cyc));
    line([x x], [y1(1) y1(2)], 'Color', 'r');
end
title('DC sub')

subplot(3,1,3)
shift_y = dc_sub + mean(rawtrace);
dF = (shift_y - mean(shift_y)) ./ mean(shift_y);
plot(y_sec(1:length(dF)), dF*100, 'k')
y1 = get(gca,'ylim');
for cyc=1:length(cycle_starts)
    x = mw_sec(cycle_starts(cyc));
    line([x x], [y1(1) y1(2)], 'Color', 'r');
end
%plot(mw_sec(cycle_starts)-vol_offset, ones(size(mw_sec(cycle_starts)))*mean(dF), 'r.');
title('dF rolled')

suptitle(sprintf('ROI %i', roi));
xlabel('time (sec)');

figname_prefix = sprintf('timeseries_%s', rois{roi});
figname = strcat(figname_prefix, '_', strrep(acquisition_struct_fn(1:end-4), '_', '-'))
saveas(fig, fullfile(figdir, figname));

saveas(fig, fullfile(figdir, [figname '.png']));

end
%%

%%

% 
% 
% %% Choose ROIs:
% 
% % avgimg_fn = fullfile(source_dir, 'AVG', ['AVG_' curr_tiff_fn]);
% %avgimg_fn = 'STD_fov2_bar5_00002.tif #1.tif #11.tif';
% avgimg_fn = 'AVG_fov2_bar5_00001.tif #1.tif #15.tif';
% avgimg = imread(fullfile(source_dir, 'slices', avgimg_fn));
% avgimg = mat2gray(avgimg);
% 
% create_masks = 0;
% 
% mask_acquisition_struct_fn = fullfile(source_dir, 'bar5_cond2_VOL_tile.mat');
% if create_masks==1
%     slice_no = 'slice15';
%     avgimg_fn = 'AVG_fov2_bar5_00001.tif #1.tif #15.tif';
%     avgimg = imread(fullfile(source_dir, 'slices', avgimg_fn));
%     avgimg = mat2gray(avgimg);
%     masks=ROIselect_circle(avgimg);
% else
%     mask_struct = load(mask_acquisition_struct_fn);
%     masks = mask_struct.D.masks;
% end
% 
% nframes = size(Y,3);
% 
% % -------------------------------------------------------------------------
% % extract raw traces:
% raw_traces = zeros(size(masks,3), size(Y,3));
% for r=1:size(masks,3)
%     curr_mask = masks(:,:,r);
%     Y_masked = nan(1,size(Y,3));
%     for t=1:size(Y,3)
%         t_masked = curr_mask.*Y(:,:,t);
%         t_masked(t_masked==0) = NaN;
%         Y_masked(t) = nanmean(t_masked(:));
%         %Y_masked(t) = sum(t_masked(:));
%     end
%     raw_traces(r,:,:) = Y_masked;
% end
% 
% %acquisition_struct_fn = sprintf('bar%i_cond%i_slice%i_tile.mat', run_idx, cond_idx, slice_idx)
% acquisition_struct_fn = sprintf('bar%i_cond%i_VOL_tile.mat', run_idx, cond_idx)
% D = struct();
% %D.slice = slice_no;
% D.condition = cond_no;
% D.run = run_no;
% D.masks = masks;
% D.avgimg = avgimg;
% D.avgimg_fn = avgimg_fn;
% D.curr_tiff = curr_tiff_fn;
% D.source_dir = source_dir;
% D.raw_traces = raw_traces;
% 
% save(fullfile(source_dir, acquisition_struct_fn), 'D')
% 
% % -------------------------------------------------------------------------
% % Save ROI nums to img:
% 
% nparts = strsplit(curr_tiff_fn, ' ');
% %figname = strcat(nparts(1), '_', slice_no, '_', run_no);
% figname = strcat(nparts(1), '_', 'VOL', '_', run_no);
% figname = strrep(figname, '_', '-')
% 
% RGBimg = zeros([size(avgimg),3]);
% RGBimg(:,:,1)=0;
% RGBimg(:,:,2)=avgimg;
% RGBimg(:,:,3)=0;
% 
% %figure()
% numcells=size(masks,3);
% for c=1:numcells
%     RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,c);
% end
% 
% fig = figure();
% imshow(RGBimg);
% title(figname);
% hold on;
% for c=1:numcells
%     [x,y] = find(masks(:,:,c)==1,1);
%     text(y,x,num2str(c))
%     hold on;
% end
% 
% D.RGBimg = RGBimg;
% save(acquisition_struct_fn, 'D')
% 
% % -------------------------------------------------------------------------
% % Save ROI map (and make figure dir if needed):
% % Use acquisition_struct_fn name to name dir for data from that struct.
% 
% figdir = fullfile(source_dir, 'figures', acquisition_struct_fn(1:end-4));
% if ~exist(figdir, 'dir')
%     mkdir(figdir)
% end
% 
% figname = strcat('ROIs - ', strrep(acquisition_struct_fn(1:end-4), '_', '-'));
% saveas(fig, fullfile(figdir, figname));
% 
% 
% %
% %%  PHASE MAP?
% 
% target_freq = 0.37;
% Fs = 5.58;
% 
% winsz = round((1/target_freq)*Fs*2);
% %nrois = size(D.raw_traces,1);
% nrois = size(raw_traces,1);
% 
% fft_struct = struct();
% for roi=1:nrois;
%     
%     roi_no = sprintf('roi%i', roi);
%     
% %     fin = ceil((5.58/.37)*30);
% %     pix = D.(slice_no).(file_no).raw_traces(roi, 1:round(fin));
% %     actualT = 0:length(pix);
% %     interpT = 0:(1/5.58):fin;
% %     tmpy = interp1(actualT, pix, interpT);
% 
%     %tmpy = D.raw_traces(roi, 1:151);
%     %tmpy = D.raw_traces(roi, 1:round(fin));
%     
%     %vol_trace = D.raw_traces(roi, :);
%     vol_trace = raw_traces(roi, :);
%     
%     for slice=15:15 %ntotal_slices;
%         slice_indices = slice:ntotal_slices:ntotal_frames;
%         vol_offsets = y_sec_vols(slice_indices);
%         
%         tmp0 = vol_trace(slice_indices);
%         tmp1 = padarray(tmp0,[0 winsz],tmp0(1),'pre');
%         tmp1 = padarray(tmp1,[0 winsz],tmp1(end),'post');
%         rollingAvg=conv(tmp1,fspecial('average',[1 winsz]),'same');%average
%         rollingAvg=rollingAvg(winsz+1:end-winsz);
%         y = tmp0 - rollingAvg;
%         
% %         y = y + mean(tmp0);
% 
% %         s1 = smooth(tmp0, winsz, 'rlowess');
% %         t1 = tmp0 - s1';
% %         t1 = t1 + mean(tmp0);
% 
%     
% %     start_slice = 15;
% %     slice_indices = start_slice:ntotal_slices:ntotal_frames;
% %     figure(); 
% %     subplot(1,2,1); plot(tmp0(slice_indices));
% %     hold on; plot(y(slice_indices));
% %     hold on; plot(t1(slice_indices));
% %     legend({'raw', 'rolling', 'smooth'})
% %     
% %     df_raw = (tmp0(slice_indices) - mean(tmp0(slice_indices))) ./ mean(tmp0(slice_indices));
% %     df_roll = (y(slice_indices) - mean(y(slice_indices))) ./ mean(y(slice_indices));
% %     df_smooth = (t1(slice_indices) - mean(t1(slice_indices))) ./ mean(t1(slice_indices));
% %     subplot(1,2,2); plot(df_raw*100); 
% %     hold on; plot(df_roll*100); hold on; plot(df_smooth*100);
% %     legend({'raw', 'rolling', 'smooth'})
% 
%     
%         NFFT = length(y);
%         fft_y = fft(y,NFFT);
%         %F = ((0:1/NFFT:1-1/NFFT)*Fs).';
%         F = Fs*(0:(NFFT/2))/NFFT;
%         freq_idx = find(abs((F-target_freq))==min(abs(F-target_freq)));
% 
%     %     figure(); plot(F(1:length(F)/2), abs(fft_y(1:length(F)/2)))
%     %     hold on; plot(F(freq_idx), max(abs(fft_y(1:length(F)/2))), 'r*');
% 
%     %     subplot(4,5,roi);
%     %     plot(F, abs(fft_y));
%     %     hold on; plot(F(freq_idx), max(abs(fft_y)), 'g*');
%     %     title(sprintf('ROI: %i', roi));
% 
%         magY = abs(fft_y);
%         %phaseY = unwrap(angle(Y));
%         phaseY = angle(fft_y);
%         %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
%         %phase_deg = phase_rad / pi * 180 + 90;
% 
%         fft_struct.(roi_no).targetPhase = phaseY(freq_idx); % unwrap(angle(Y(freq_idx)));
%         fft_struct.(roi_no).targetMag = magY(freq_idx);
%         
%         fft_struct.(roi_no).fft_y = fft_y;
%         fft_struct.(roi_no).DC_y = y;
%         fft_struct.(roi_no).raw_y = tmp0;
%         fft_struct.(roi_no).slices = slice_indices;
%         fft_struct.(roi_no).freqs = F;
%         fft_struct.(roi_no).freq_idx = freq_idx;
%         fft_struct.(roi_no).target_freq = target_freq;
%         
%     end
%     
% end
% 
% 
% % -------------------------------------------------------------------------
% % Look at power spectra:
% 
% plot_semilog = 0
% fig = figure();
% rois = fieldnames(fft_struct);
% for roi=1:length(rois)
%     subplot(3,3,roi)
%     curr_fft = fft_struct.(rois{roi}).fft_y;
%     L = length(curr_fft);
%     mags = abs(curr_fft) / L;
%     power = mags.^2;
%     freqs = fft_struct.(rois{roi}).freqs;
%     freq_idx = fft_struct.(rois{roi}).freq_idx;
%     
%     P1 = power(1:L/2+1);
%     %P1(2:end-1) = 2*P1(2:end-1);
%     
%     if plot_semilog==1
%         %semilogx(fax_Hz(1:N_2), 20*log10(X_mags(1:N_2)))
%         semilogx(freqs, 20*log10(mags(1:L/2+1)), 'k')
%         hold on
%         %semilogx(freqs(freq_idx), 20*log10(mags(freq_idx)), 'r*')
%         semilogx(freqs(freq_idx), max(20*log10(mags)), 'r*')
%         ylabel('power (dB)')
%         xlabel('F (Hz)')
%         figname_prefix = 'power-dB';
%     else
%         plot(freqs, P1, 'k')
%         hold on
%         plot(freqs(freq_idx), max(P1), 'r*')
%         ylabel('power')
%         xlabel('F (Hz)')
%         figname_prefix = 'power';
%     end
%     title(rois{roi})
% end
% 
% plotname = strrep(acquisition_struct_fn(1:end-4), '_', '-');
% suptitle(plotname)
% 
% figname = strcat(figname_prefix, '_', strrep(acquisition_struct_fn(1:end-4), '_', '-'))
% saveas(fig, fullfile(figdir, figname));
% 
% 
% % -------------------------------------------------------------------------
% % Look at PHASE map by ROIs:
% phaseMap = zeros(size(avgimg));
% phaseMap(phaseMap==0) = NaN;
% for roi=1:length(fieldnames(fft_struct))
%     
%     roi_no = sprintf('roi%i', roi);
%     curr_phase = fft_struct.(roi_no).targetPhase;
%     curr_mag = fft_struct.(roi_no).targetMag;
%     replaceNan = masks(:,:,roi)==1;
%     phaseMap(replaceNan) = 0;
%     phaseMap = phaseMap + curr_phase*masks(:,:,roi);
%     
% end
% 
% fig = figure();
% acquisition = repmat(avgimg, [1, 1, 3]);
% B = phaseMap;
% subplot(1,2,1)
% imagesc(acquisition);
% hold on;
% Bimg = imagesc2(B);
% colormap hsv
% caxis([-pi, pi])
% %colorbar()
% 
% % legend:
% 
% figname = strcat(figname_prefix, '_', strrep(acquisition_struct_fn(1:end-4), '_', '-'))
% saveas(fig, fullfile(figdir, figname));
% 
% 
% %% PLot time series with stimulus onsets:
% 
% roi = 6;
% 
% %vol_offset = vol_offsets(15);
% rawtrace = fft_struct.(rois{roi}).raw_y;
% dc_sub = fft_struct.(rois{roi}).DC_y;
% 
% 
% figure(); 
% subplot(3,1,1)
% plot(y_sec, rawtrace, 'k'); hold all;
% %plot(mw_sec(cycle_starts)-vol_offsets(15), ones(size(mw_sec(cycle_starts)))*mean(rawslice), 'r.')
% y1 = get(gca,'ylim');
% for cyc=1:length(cycle_starts)
%     x = mw_sec(cycle_starts(cyc));
%     line([x x], [y1(1) y1(2)], 'Color', 'r');
% end
% title('raw')
% 
% subplot(3,1,2)
% plot(y_sec, dc_sub, 'k')
% %plot(mw_sec(cycle_starts)-vol_offset, ones(size(mw_sec(cycle_starts)))*mean(y), 'r.');
% y1 = get(gca,'ylim');
% for cyc=1:length(cycle_starts)
%     x = mw_sec(cycle_starts(cyc));
%     line([x x], [y1(1) y1(2)], 'Color', 'r');
% end
% title('DC sub')
% 
% subplot(3,1,3)
% shift_y = dc_sub + mean(rawtrace);
% dF = (shift_y - mean(shift_y)) ./ mean(shift_y);
% plot(y_sec, dF*100, 'k')
% y1 = get(gca,'ylim');
% for cyc=1:length(cycle_starts)
%     x = mw_sec(cycle_starts(cyc));
%     line([x x], [y1(1) y1(2)], 'Color', 'r');
% end
% %plot(mw_sec(cycle_starts)-vol_offset, ones(size(mw_sec(cycle_starts)))*mean(dF), 'r.');
% title('dF rolled')
% 
% suptitle('ROI 6');
% xlabel('time (sec)');
% 
% 
% 
% %%
% % 
% % % -------------------------------------------------------------------------
% % % 2.  Split channels (if no motion correction, raw data have Ch1/Ch2
% % % interleaved):
% %     
% % 
% %     %S = load(strcat(source_dir, 'mw_data/', pymat_fn));
% %     %S = load(pymat_fn_ard200);
% %     
% %     % Get MW-specific info: -----------------------------------------------
% %     mw_times = S.mw_times_by_file;
% %     offsets = double(S.offsets_by_file);
% %     mw_codes = S.mw_codes_by_file;
% %     mw_file_durs = double(S.mw_file_durs);
% %         
% %     mw_file_no = sprintf('mw%i%s', mw_fidx, run_no);
% %     if strcmp(class(mw_times), 'int64')
% %         curr_mw_times = double(mw_times(mw_fidx, :));                       % GET MW rel times from match_triggers.py: 'rel_times'
% %         curr_mw_codes = mw_codes(mw_fidx, :);                               % Get MW codes corresponding to time points:
% %         fprintf('File %s, Last code found: %i\n', mw_file_no, curr_mw_codes(end-1))
% %     else
% %         curr_mw_times = double(mw_times{mw_fidx}); % convert to sec
% %         curr_mw_codes = mw_codes{mw_fidx};
% %     end
% %     curr_mw_times(end+1) = double(S.stop_ev_time);
% %     mw_rel_times = ((curr_mw_times - curr_mw_times(1)) + offsets(mw_fidx)); % shift to have 0 be start
% %     mw_sec = mw_rel_times/1000000;                                          % into seconds
% %     
% %     if isfield(S, 'acquisition_evs')
% %         si_onsets = double(S.frame_onset_times);
% %         si_offsets = double(S.frame_offset_times);
% %     end
% %     si_times = [si_onsets; si_offsets];
% %     si_times = si_times(:).';
% %     
% %     y_sec = (si_times - si_times(1)) / 1000000;   
% %     
% %     if length(si_onsets) ~= nframes
% %         fprintf('Missing %i SI frame triggers in %s.\n', (nframes - length(si_onsets)), pymat_fn);
% %         interpolate_y = 1;
% %     else
% %         fprintf('SI frame triggers match n frames. Not interpolating.\n');
% %     end
% %     
% %     
% % %%
% % avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
% % ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
% % %stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);
% % 
% % slice_no = ['slice' num2str(slice_idx)];
% % file_no = ['file' num2str(run_idx)]; % 'file1R'
% % % file_no = ['code' num2str(code_idx)]; % 'file1R'
% % % -------------------------------------------------------------------------
% % 
% % parts = strsplit(source_dir, '/');
% % cond_folder = parts{end-1};
% % 
% % % Set dir for new (or to-be-appended struct):
% % acquisition_struct_fn = strcat(source_dir, slice_no, '_traces_AVG.mat')
% % 
% % 
% 
% 
% %%  PHASE MAP?
% 
% target_freq = 0.37;
% Fs = 5.58;
% 
% avgimg = D.(slice_no).(file_no).avgimg;
% masks = D.(slice_no).(file_no).masks;
% 
% winsz = 30;
% for roi=1:size(D.(slice_no).(file_no).raw_traces,1)
%     
%     roi_no = sprintf('roi%i', roi);
%     
%     fin = round((5.58/.37)*30);
% %     pix = D.(slice_no).(file_no).raw_traces(roi, 1:round(fin));
% %     actualT = 0:length(pix);
% %     interpT = 0:(1/5.58):fin;
% %     tmpy = interp1(actualT, pix, interpT);
%     
%     tmpy = D.(slice_no).(file_no).raw_traces(roi, 1:round(fin));
%     %tmpy = D.(slice_no).(file_no).raw_traces(roi, 1:end);
%     tmp1 = padarray(tmpy,[0 winsz],tmpy(1),'pre');
%     tmp1 = padarray(tmp1,[0 winsz],tmpy(end),'post');
%     rollingAvg=conv(tmp1,fspecial('average',[1 winsz]),'same');%average
%     rollingAvg=rollingAvg(winsz+1:end-winsz);
%     y = tmpy - rollingAvg;
%     
%     NFFT = length(y);
%     Y = fft(y,NFFT);
%     F = ((0:1/NFFT:1-1/NFFT)*Fs).';
%     freq_idx = find(abs((F-target_freq))==min(abs(F-target_freq)));
% 
%     magY = abs(Y);
%     %phaseY = unwrap(angle(Y));
%     phaseY = angle(Y);
%     %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
%     %phase_deg = phase_rad / pi * 180 + 90;
%     
%     fft_struct.(roi_no).targetPhase = phaseY(freq_idx); % unwrap(angle(Y(freq_idx)));
%     fft_struct.(roi_no).targetMag = magY(freq_idx);
%     
% end
% 
% phaseMap = zeros(size(avgimg));
% phaseMap(phaseMap==0) = NaN;
% for roi=1:length(fieldnames(fft_struct))
%     
%     roi_no = sprintf('roi%i', roi);
%     curr_phase = fft_struct.(roi_no).targetPhase;
%     curr_mag = fft_struct.(roi_no).targetMag;
%     replaceNan = masks(:,:,roi)==1;
%     phaseMap(replaceNan) = 0;
%     phaseMap = phaseMap + curr_phase*masks(:,:,roi);
%     
% end
% 
% figure()
% acquisition = repmat(avgimg, [1, 1, 3]);
% B = phaseMap;
% imagesc(acquisition);
% hold on;
% Bimg = imagesc2(B);
% colormap hsv
% caxis([-pi, pi])
% %colorbar()
% 
% %% legend?
% 
% sim_path = '/home/juliana/Repositories/retinotopy-mapper/tests/simulation/';
% sim_fn = 'V-Left_0_8bit.tif';
% 
% sframe = 1;
% SIM = bigread2(strcat(sim_path, sim_fn),sframe);
% if ~isa(SIM,'double');    SIM = double(SIM);  end
% nframes = size(SIM,3);
% 
% Fs = 60;
% T = 1/Fs;
% L = nframes;
% t = (0:L-1)*T;
% target = 0.05;
% legend_phase = zeros(size(SIM,1), size(SIM,2));
% for x=1:size(SIM,1)
%     for y=1:size(SIM,2)
%         pix = SIM(x,y,:);
%         NFFT = length(pix);
%         Yf = fft(pix,NFFT);
%         F = ((0:1/NFFT:1-1/NFFT)*Fs).';
%         freq_idx = find(abs((F-target))==min(abs(F-target)));
% 
%         %magY = abs(Yf);
%         %phaseY = unwrap(angle(Y));
%         %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
%         %phase_deg = phase_rad / pi * 180 + 90;
% 
%         legend_phase(x,y) = angle(Yf(freq_idx)); % unwrap(angle(Y(freq_idx)));
%     end
%         
% end
% figure()
% imagesc(legend_phase)
% colormap hsv
% caxis([-pi, pi])
% colorbar()
% 
% 
% %% fake legend:
% 
% acquisition = zeros(10,50);
% acquisition(:,1) = ones(size(acquisition(:,1)));
% legend_im = zeros(10,50,1000);
% legend_im(:,:,1) = acquisition(:,:,1);
% tmpA = acquisition;
% for lidx=2:1000
%     legend_im(:,:,lidx) = circshift(tmpA, 1, 2);
%     tmpA = legend_im(:,:,lidx);
% end
% 
% Fs = 1;
% T = 1/Fs;
% L = size(legend_im,3);
% t = (0:L-1)*T;
% target_freq = 1/50;
% legend_phase = zeros(size(legend_im,1), size(legend_im,2));
% for x=1:size(legend_im,1)
%     for y=1:size(legend_im,2)
%         pix = legend_im(x,y,:);
%         NFFT = length(pix);
%         Y = fft(pix,NFFT);
%         F = ((0:1/NFFT:1-1/NFFT)*Fs).';
%         freq_idx = find(abs((F-target_freq))==min(abs(F-target_freq)));
% 
%         magY = abs(Y);
%         %phaseY = unwrap(angle(Y));
%         %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
%         %phase_deg = phase_rad / pi * 180 + 90;
% 
%         legend_phase(x,y) = angle(Y(freq_idx)); % unwrap(angle(Y(freq_idx)));
%     end
%         
% end
% figure()
% imagesc(legend_phase)
% colormap jet
% caxis([-pi, pi])
% %colorbar()
% axis('off')
% %helperFrequencyAnalysisPlot1(F, magY, phaseY, NFFT)
% 
% %% How many active cells have matching freq?
% 
% target_freq = 0.37;
% matchingROIs = [];
% for roi=1:length(active_cells)
%     x = D_traces(D_active(roi), :);
%     
%     
%     psdest = psd(spectrum.periodogram,x,'Fs',Fs,'NFFT',length(x));
%     [~,I] = max(psdest.Data);
%     fprintf('Maximum occurs at %d Hz.\n',psdest.Frequencies(I));
%     
%     if sprintf('%0.2f', psdest.Frequencies(I))==num2str(target_freq)
%         matchingROIs = [matchingROIs D_active(roi)];
%     end
% 
% end
% 
% fprintf('Found %i out of %i total ROIs with peak at target frequency of %0.2f', length(matchingROIs), length(active_cells), target_freq)
% 
