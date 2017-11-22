
% DEFINE SOURCE:
% run_idx = 10;
% slice_idx = 15;
% traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/CC/';
% tiff_path = sprintf('/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/run%i_slices/', run_idx);
% curr_tiff = sprintf('fov1_gratings_000%02d.tif #2.tif #%i.tif', run_idx, slice_idx);
% curr_tiff_path = strcat(tiff_path, curr_tiff);

%% GCaMP:

%/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/fov2_bar5_00002/ch1_slices
run_idx = 2;
slice_idx = 10;
%traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/CC/';
tiff_path = sprintf('/nas/volume1/2photon/RESDATA/TEFO/20161218_CE024/fov2_bar5_0000%i/ch1_slices/', run_idx);
s = strsplit(tiff_path, 'run');
source_path = s(1);
curr_tiff = sprintf('fov2_bar5_000%02d.tif #1.tif #%i.tif', run_idx, slice_idx);
curr_tiff_path = strcat(tiff_path, curr_tiff);
frame_rate = 145.13; %156.13;
%

run_idx = 1;
slice_idx = 10;
tiff_path = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/rsvp/fov6_rsvp_bluemask_test_10trials/fov6_rsvp_bluemask_test_10trials_00001/ch1_slices/';
s = strsplit(tiff_path, 'run');
source_path = s(1);
curr_tiff = sprintf('fov6_rsvp_bluemask_test_10trials_000%02d.tif #1.tif #%i.tif', run_idx, slice_idx);
curr_tiff_path = strcat(tiff_path, curr_tiff);
frame_rate = 145.13; %Hz; %156.13;

%% NOISE:
% single plane:
run_idx = 1;
slice_idx = 10;
traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/CC/';
tiff_path = sprintf('/nas/volume1/2photon/RESDATA/TEFO/20160115_AG33/fov3_gratings1/run%i_slices/', run_idx);
s = strsplit(tiff_path, 'run');
source_path = s(1);
curr_tiff = sprintf('fov3_gratings_000%02d.tif #2.tif #%i.tif', run_idx, slice_idx);
curr_tiff_path = strcat(tiff_path, curr_tiff);

% single volume:
curr_tiff = 'VOL1_fov3_gratings_00001_CH2.tif';
curr_tiff_path = char(strcat(source_path, curr_tiff));

%% Less noise:
% TRY FILTERING:
run_idx = 1;
slice_idx = 11; %22;
traces_path = '/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/CC/';
tiff_path = sprintf('/nas/volume1/2photon/RESDATA/TEFO/20160118_AG33/fov1_gratings1/run%i_slices/', run_idx);
s = strsplit(tiff_path, 'run');
source_path = char(s(1));
curr_tiff = sprintf('fov1_gratings_000%02d.tif #2.tif #%i.tif', run_idx, slice_idx);
curr_tiff_path = strcat(tiff_path, curr_tiff);

filtered_dir = char(sprintf('run%i_slice%i_filtered/', run_idx, slice_idx));
filtered_path = char(strcat(source_path, filtered_dir))

if ~exist(filtered_path, 'dir')
    mkdir(filtered_path);
end
%% Read in tiff:

sframe = 1;
Y = bigread2(curr_tiff_path,sframe);

%Y = Y - min(Y(:)); 
if ~isa(Y,'double');    Y = double(Y);  end         % convert to single

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

% 
% [cc]=CrossCorrImage(Y);
% cc_fn = strcat(traces_path, 'cc_', curr_tiff(1:end-4), '.png');
% imwrite(cc, cc_fn);

%

%% FILTERS?

%%


cutoff = 1e3 %1e5; % 100kHz
Fnorm = cutoff/(Fs/2);           % Normalized frequency
df = designfilt('lowpassfir','FilterOrder',70,'CutoffFrequency',Fnorm);
t = (0:L-1)/Fs;              % time vector
D = mean(grpdelay(df))


newtiff_name = char(strcat(filtered_path, curr_tiff, '.tif'));

YY = zeros(size(Y));
for frame_no=1:size(Y,3)
    curr_frame = Y(:,:,frame_no);
    for row=2:2:size(curr_frame,1)
        curr_frame(row,:) = fliplr(curr_frame(row,:));      % flip for BiDi scanning 
    end
    unraveled = reshape(permute(curr_frame, [2 1]), [1,d]); % switch col/rows for reshaping by 'ROW' (reshape works column-wise)
    
    %fx = filter(Hd, unraveled);                             % low-pass filter 
    fx = filter(df,[unraveled'; zeros(D,1)]);   % Append D zeros to the input data
    fx = fx(D+1:end)';                          % Shift data to compensate for delay

    
    reraveled = permute(reshape(fx, [d1 d2]), [2 1]);       % restore row/col switch & reshape frame
    for rrow=2:2:size(reraveled,1)
       reraveled(rrow,:) = fliplr(reraveled(rrow,:));       % undo l/r flip
    end
    YY(:,:,frame_no) = reraveled;
    
%     if frame_no==1
%         imwrite(uint16(YY), newtiff_name);
%     else
%         imwrite(uint16(YY), newtiff_name, 'WriteMode', 'append');
%     end
    
end

figure()
subplot(1,2,1)
imshow(Y(:,:,frame_no));
hold on;
subplot(1,2,2)
imshow(YY(:,:,frame_no));
colorbar()


% 
YY = uint16(YY);

% t = Tiff(curr_tiff_path,'r');
% imageData = read(t);
% close(t)

tinfo = imfinfo(curr_tiff_path);
for fidx=1:size(YY,3)
newtiff_name = char(strcat(filtered_path, curr_tiff, sprintf(' filtered%i.tif', fidx)));
t = Tiff(newtiff_name, 'w8');

t.setTag('ImageLength', tinfo(1).Height);
t.setTag('ImageWidth', tinfo(1).Width);
t.setTag('Photometric', 1); %tinfo(1).PhotometricInterpretation); %Tiff.Photometric.RGB);
t.setTag('BitsPerSample', tinfo(1).BitsPerSample);
t.setTag('SamplesPerPixel', 1); %size(YY,3)); %tinfo(1).SamplesPerPixel);
%t.setTag('TileWidth',  16); %tinfo(1).TileWidth);
%t.setTag('TileLength', 16); %tinfo(1).TileWidth);
t.setTag('Compression', 1); %tinfo(1).Compression); %Tiff.Compression.JPEG);
t.setTag('PlanarConfiguration', 1); %tinfo(1).PlanarConfiguration); %Tiff.PlanarConfiguration.Chunky);
t.setTag('Software', 'MATLAB');

t.write(YY(:,:,fidx));
t.close();
end
% imwrite(YY, char(strcat(filtered_path, curr_tiff, '.tif')), 'tif')
%     
% imwrite(image1, 'foobar.tif');
% imwrite(image2, 'foobar.tif', 'WriteMode', 'append');
% imwrite(image3, 'foobar.tif', 'WriteMode', 'append');


%% Plot single-sided power spectrum (averaged):

fig_prefix = 'AVG_power';

do_filter = 1;
winsize = 1440;

%frame_rate = 145.13; %Hz; %156.13;
Fs = 120*120*frame_rate;
fVals = 0:Fs/L:Fs/2;

Px_mat = zeros(size(Y,3), size(fVals,2));
for frame_no=1:size(Y,3)
    curr_frame = Y(:,:,frame_no);
    for row=2:2:size(curr_frame,1)
        curr_frame(row,:) = fliplr(curr_frame(row,:));
    end
    unraveled = reshape(permute(curr_frame, [2 1]), [1,d]);
    if do_filter==1
        tmp1=padarray(unraveled,[0 winsize],unraveled(1),'pre');
        tmp1=padarray(tmp1,[0 winsize],unraveled(end),'post');
        rollingAvg=conv(tmp1,fspecial('average',[1 winsize]),'same');%average
        rollingAvg=rollingAvg(winsize+1:end-winsize);
        signal = unraveled - rollingAvg;
    else
        signal = unraveled;
    end
    
    L = length(signal);
    xdft = fft(signal);
    Px = abs(xdft(1:length(signal)/2+1)).^2;
    
    Px_mat(frame_no,:) = Px;
    
end

Px_avg = mean(Px_mat, 1);

% PLOT:
% -------------------------------------------------------------------
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
fig = figure();
plot(fVals, Px_avg,'b','LineSmoothing','on','LineWidth',1);	
%plot(fVals, Px_avg,'b','LineSmoothing','on','LineWidth',1);	
hold on;
[max_val,max_idx] = max(Px_avg);

plot(fVals(max_idx), Px_avg(max_idx), 'r*');
ylabel('Power');
max_str = sprintf('Max: %d Hz.\n',fVals(max_idx));
title(max_str, 'FontSize', 12)

figname = strcat(source_path, strcat(fig_prefix, sprintf('_%s.png', curr_tiff)));
% set(gcf,'units','normalized','outerposition',[0 0 1 1])
%set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig, 'PaperPositionMode', 'auto')
saveas(fig, char(figname));


[sortedX,sortingIndices] = sort(Px_avg,'descend');
for i=1:10
    %fprintf('Rank %i power occurs at %d Hz.\n', i, fVals(sortingIndices(i))); 
    fprintf('Rank %i: %d Hz.\n', i, fVals(sortingIndices(i))); 
end


%% Plot single-sided power spectrum (averaged):

fig_prefix = 'AVG_power_dB';

do_filter = 1;
winsize = 1440;

frame_rate = 156.13;
Fs = 120*120*frame_rate;
fVals = 0:Fs/L:Fs/2;
L = d1*d2;

Px_mat = zeros(size(Y,3), size(fVals,2));
for frame_no=1:size(Y,3)
    curr_frame = Y(:,:,frame_no);
    for row=2:2:size(curr_frame,1)
        curr_frame(row,:) = fliplr(curr_frame(row,:));
    end
    unraveled = reshape(permute(curr_frame, [2 1]), [1,d]);
    if do_filter==1
        tmp1=padarray(unraveled,[0 winsize],unraveled(1),'pre');
        tmp1=padarray(tmp1,[0 winsize],unraveled(end),'post');
        rollingAvg=conv(tmp1,fspecial('average',[1 winsize]),'same');%average
        rollingAvg=rollingAvg(winsize+1:end-winsize);
        signal = unraveled - rollingAvg;
    else
        signal = unraveled;
    end
    
    L = length(signal);
    xdft = fft(signal);
    Px = 1/(L*Fs)*abs(xdft(1:length(signal)/2+1)).^2;
    
    Px_mat(frame_no,:) = Px;
    
end

Px_avg = mean(Px_mat, 1);

% PLOT:
% -------------------------------------------------------------------
% fig = figure('units','normalized','outerposition',[0 0 1 1]);
fig = figure();
plot(fVals, 10*log10(Px_avg),'b','LineSmoothing','on','LineWidth',1);	
%plot(fVals, Px_avg,'b','LineSmoothing','on','LineWidth',1);	
hold on;
[max_val,max_idx] = max(Px_avg);

plot(fVals(max_idx), 10*log10(Px_avg(max_idx)), 'r*');
ylabel('dB/Hz');
max_str = sprintf('Max: %d Hz.\n',fVals(max_idx));
title(max_str, 'FontSize', 12)


figname = strcat(source_path, strcat(fig_prefix, sprintf('_%s.png', curr_tiff)));
% set(gcf,'units','normalized','outerposition',[0 0 1 1])
%set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig, 'PaperPositionMode', 'auto')
saveas(fig, char(figname));


[sortedX,sortingIndices] = sort(Px_avg,'descend');
for i=1:10
    fprintf('Rank %i power occurs at %d Hz.\n', i, fVals(sortingIndices(i))); 
end


%%


Fpass = 150e3/(Fs/2);
Fstop = 300e3/(Fs/2);
Ap = 1;
Ast = 30;

d = designfilt('lowpassfir','PassbandFrequency',Fpass,...
  'StopbandFrequency',Fstop,'PassbandRipple',Ap,'StopbandAttenuation',Ast);

hfvt = fvtool(d);
N = filtord(d)

dk = designfilt('lowpassfir','PassbandFrequency',Fpass,...
  'StopbandFrequency',Fstop,'PassbandRipple',Ap,...
  'StopbandAttenuation',Ast, 'DesignMethod', 'kaiserwin');

addfilter(hfvt,dk);
legend(hfvt,'Equiripple design', 'Kaiser window design')

N = filtord(dk)



Fp = 100e3;
Fst = 300e3;
Ap = 1;
Ast = 60;

dbutter = designfilt('lowpassiir','PassbandFrequency',Fp,...
  'StopbandFrequency',Fst,'PassbandRipple',Ap,...
  'StopbandAttenuation',Ast,'SampleRate',Fs,'DesignMethod','butter');

hfvt = fvtool(dbutter,'Analysis','grpdelay');

%% Compensating for Constant Filter Delay:

x = signal;

cutoff = 1e3 %1e5; % 100kHz
Fnorm = cutoff/(Fs/2);           % Normalized frequency
df = designfilt('lowpassfir','FilterOrder',70,'CutoffFrequency',Fnorm);
t = (0:L-1)/Fs;              % time vector

grpdelay(df,2048,Fs)   % plot group delay
D = mean(grpdelay(df)) % filter delay in samples

y = filter(df,[x'; zeros(D,1)]); % Append D zeros to the input data
y = y(D+1:end);                  % Shift data to compensate for delay

figure
plot(t,x,t,y,'r','linewidth',1.5);
title('Filtered Waveforms');
xlabel('Time (s)')
legend('Original Noisy Signal','Filtered Signal');
grid on
axis tight


% Design a 7th order lowpass IIR elliptic filter with cutoff frequency
% of 75 Hz.
Fnorm = cutoff/(Fs/2); % Normalized frequency
df = designfilt('lowpassiir',...
               'PassbandFrequency',Fnorm,...
               'FilterOrder',7,...
               'PassbandRipple',1,...
               'StopbandAttenuation',30);
           
% Plot the group delay of the filter and notice that it varies with 
% frequency indicating that the filter delay is frequency-dependent.
grpdelay(df,2048,'half',Fs)

% Filters that introduce constant delay are linear phase filters. Filters 
% that introduce frequency-dependent delay are non-linear phase filters.
% Filter the data and look at the effects of each filter implementation on 
% the time signal.

y1 = filter(df,x);    % non-linear phase filter - no delay compensation
y2 = filtfilt(df,x);  % zero-phase implementation - delay compensation

figure
plot(t,x);
hold on
plot(t,y1,'r','linewidth',1.5);
plot(t,y2,'g','linewidth',1.5);
title('Filtered Waveforms');
xlabel('Time (s)')
legend('Original Signal','Non-linear phase IIR output',...
  'Zero-phase IIR output');
ax = axis;
%axis([0.25 0.55 ax(3:4)])
grid on


%% Check how filtered frame looks:
reraveled = permute(reshape(unraveled, [d1 d2]), [2 1]);       % restore row/col switch & reshape frame
hpass = permute(reshape(signal, [d1 d2]), [2 1]);
lpass = permute(reshape(y, [d1 d2]), [2 1]);
for rrow=2:2:size(reraveled,1)
   reraveled(rrow,:) = fliplr(reraveled(rrow,:));       % undo l/r flip
   hpass(rrow,:) = fliplr(hpass(rrow,:));
   lpass(rrow,:) = fliplr(lpass(rrow,:));
end
%YY(:,:,frame_no) = reraveled;
figure()
subplot(2,2,1)
imshow(Y(:,:, frame_no));
title('original');
cmin = min(int16(Y(:)));
cmax = max(int16(Y(:)));
caxis([cmin cmax])
colorbar()
hold on;
subplot(2,2,2)
imshow(reraveled)
title('reconstr.');
cmin = min(int16(reraveled(:)));
cmax = max(int16(reraveled(:)));
caxis([cmin cmax])
colorbar()
hold on;
subplot(2,2,3)
imshow(int16(hpass));
title('HP, ravg')
cmin = min(int16(hpass(:)));
cmax = max(int16(hpass(:)));
caxis([cmin cmax])
colorbar()
hold on;
subplot(2,2,4)
imshow(int16(lpass));
title('LP filt.')
cmin = min(int16(lpass(:)));
cmax = max(int16(lpass(:)));
caxis([cmin cmax])
colorbar()

%% [Pxx,F] = pwelch(X,WINDOW,NOVERLAP,NFFT,Fs);

[P,F] = pwelch(unraveled,ones(L,1),L/2,L,Fs,'power');
helperFilterIntroductionPlot1(F,P,[fVals(max_idx) fVals(max_idx)],[0 0],...
  {'Original signal power spectrum', sprintf('%2.2f kHz Tone', fVals(max_idx)/1000)})

Fp = 1e3;    % Passband frequency in Hz
Fst = 2e3; % Stopband frequency in Hz
Ap = 1;      % Passband ripple in dB
Ast = 95;    % Stopband attenuation in dB

% Design the filter
df = designfilt('lowpassfir','PassbandFrequency',Fp,...
                'StopbandFrequency',Fst,'PassbandRipple',Ap,...
                'StopbandAttenuation',Ast,'SampleRate',Fs);

% Analyze the filter response
hfvt = fvtool(df,'Fs',Fs,'FrequencyScale','log',...
  'FrequencyRange','Specify freq. vector','FrequencyVector',F);

%%
plotPSD = 0;

oneside = 1;
do_filter = 1;
fig = figure('units','normalized','outerposition',[0 0 1 1]);

plot_no = 1;
for frame_no=1:100:size(Y,3)
    curr_frame = Y(:,:,frame_no);
    for row=2:2:size(curr_frame,1)
        curr_frame(row,:) = fliplr(curr_frame(row,:));
    end
    unraveled = reshape(permute(curr_frame, [2 1]), [1,d]);

    % Filter:
    winsize = 144; %1440; 
    if do_filter==1
        tmp1=padarray(unraveled,[0 winsize],unraveled(1),'pre');
        tmp1=padarray(tmp1,[0 winsize],unraveled(end),'post');
        rollingAvg=conv(tmp1,fspecial('average',[1 winsize]),'same');%average
        rollingAvg=rollingAvg(winsize+1:end-winsize);
        signal = unraveled - rollingAvg;
    else
        signal = unraveled;
    end

    frame_rate = 156.13;
    Fs = 120*120*frame_rate;
    L = length(signal);
    NFFT = length(signal);
    
    
    %ft = fftshift(fft(signal,n));
    %Px=ft.*conj(ft)/(NFFT*L); %Power of each freq components	
    %ft = fft(signal);
    %Px = abs(ft/L);

    
    subplot(5,4,plot_no)
    
    if oneside==1
%         if plotPSD==1
%             ft = fft(signal);
%             ft = fft(1:L/2+1);
%             psdx = (1/(Fs*L)) * abs(ft).^2;
%             psdx(2:end-1) = 2*psdx(2:end-1);
%             fVals = 0:Fs/length(signal):Fs/2;
%             plot(fVals, 10*log10(psdx))
%             hold on
%             grid on
%             title('Periodogram Using FFT')
%             xlabel('Frequency (Hz)')
%             ylabel('Power/Frequency (dB/Hz)')
%             pdg = 10*log10(psdx);
%             [max_val, max_idx] = max(pdg);
%             plot(fVals(max_idx), pdg(max_idx), 'r*');
%             [sortedX,sortingIndices] = sort(pdg,'descend');
%         else
        %fVals=Fs*(0:NFFT/2-1)/NFFT;	 % one-sided
            xdft = fft(signal);
            %fVals = Fs*(0:(L/2))/L;
            %Px = abs(xdft/L).^2;
            fVals = 0:Fs/L:Fs/2;
            %Px2 = Px(1:L/2+1); %Px2 = Px(1:NFFT/2);
            %Px2(2:end-1) = 2*Px2(2:end-1);
            
            Px2 = 1/(L*Fs)*abs(xdft(1:length(signal)/2+1)).^2;

            plot(fVals, 10*log10(Px2),'b','LineSmoothing','on','LineWidth',1);	
            hold on;
            [max_val,max_idx] = max(Px2);
            plot(fVals(max_idx), 10*log10(Px2(max_idx)), 'r*');
            [sortedX,sortingIndices] = sort(Px2,'descend');
            ylabel('dB/Hz');
        %end
    else
        %fVals=Fs*(-NFFT/2:NFFT/2-1)/NFFT;
        ft = fftshift(fft(signal));
        dF = Fs/L;
        fVals = -Fs/2:dF:Fs/2-dF;
        Px = abs(ft);
        %fVals = Fs*(-L/2:(L)-1)/L;
        plot(fVals,Px,'b','LineSmoothing','on','LineWidth',1);	
        hold on;
        [max_val,max_idx] = max(Px);
        plot(fVals(max_idx), Px(max_idx), 'r*');
        [sortedX,sortingIndices] = sort(Px,'descend');
    end
    ylabel_name = 'Power';
    figY = 'power';
    
    max_str = sprintf('Max: %d Hz.\n',fVals(max_idx));
    
    title(max_str, 'FontSize', 12)
    if plot_no==17
        xlabel('Frequency (Hz)')	 	 
    	ylabel(ylabel_name);
    end
    plot_no = plot_no + 1;
    
    
    
    fprintf('----------------------------------------------------------\n');
    fprintf('%s - slice %i\n', curr_tiff, slice_idx);
    fprintf('----------------------------------------------------------\n');
    %[sortedX,sortingIndices] = sort(Px(1:NFFT/2),'descend');
    %[sortedX,sortingIndices] = sort(Px2,'descend');
    for i=1:10
        fprintf('Rank %i power occurs at %d Hz.\n', i, fVals(sortingIndices(i))); 
    end



end
suptitle(curr_tiff)
figname = strcat(source_path, strcat(figY, sprintf('_run%i_slice%i.png', run_idx, slice_idx)));
% set(gcf,'units','normalized','outerposition',[0 0 1 1])
set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig, 'PaperPositionMode', 'auto')
saveas(fig, char(figname));

% print -dtif -r 150 test.tiff
% print(gcf, char(figname));

% [sortedX,sortingIndices] = sort(Px(1:NFFT/2),'descend');
% for i=1:10
%     fprintf('Rank %i power occurs at %d Hz.\n', i, fVals(sortingIndices(i))); 
% end


%%
plotPSD = 0;
fig = figure('units','normalized','outerposition',[0 0 1 1])
plot_no = 1;
for frame_no=1:100:size(Y,3)
    curr_frame = Y(:,:,frame_no);
    for row=2:2:size(curr_frame,1)
        curr_frame(row,:) = fliplr(curr_frame(row,:));
    end
    unraveled = reshape(curr_frame, [1,d]);

    % Filter:
    winsize = 1440; 

%     tmp1=padarray(unraveled,[0 winsize],unraveled(1),'pre');
%     tmp1=padarray(tmp1,[0 winsize],unraveled(end),'post');
%     rollingAvg=conv(tmp1,fspecial('average',[1 winsize]),'same');%average
%     rollingAvg=rollingAvg(winsize+1:end-winsize);
%     signal = unraveled - rollingAvg;
    signal = unraveled;

    frame_rate = 156.13;
    Fs = 120*120*frame_rate;
    L = length(signal);
    NFFT = length(signal);
    
    ft = fftshift(fft(signal,NFFT));
    Px=ft.*conj(ft)/(NFFT*L); %Power of each freq components	

    fVals=Fs*(-NFFT/2:NFFT/2-1)/NFFT;
    [max_val,max_idx] = max(Px);
    fprintf('Maximum occurs at %d Hz.\n',fVals(max_idx));
    
    
    subplot(5,4,plot_no)
    %plot(fVals,Px(1:NFFT/2),'b','LineSmoothing','on','LineWidth',1);	
    if plotPSD==1
        fVals=Fs*(-NFFT/2:NFFT/2-1)/NFFT;
        plot(fVals,10*log10(Px),'b');	 	 
        [max_val,max_idx] = max(10*log10(Px(1:NFFT/2)));
        ylabel_name = 'PSD (dB)';
        figY = 'PSD';
    else
        %fVals=Fs*(0:NFFT/2-1)/NFFT;	 % one-sided
        %plot(fVals,Px(1:NFFT/2),'b','LineSmoothing','on','LineWidth',1);	
        fVals=Fs*(-NFFT/2:NFFT/2-1)/NFFT;
        plot(fVals,Px,'b','LineSmoothing','on','LineWidth',1);	
        [max_val,max_idx] = max(Px(1:NFFT/2));
        ylabel_name = 'Power';
        figY = 'power';
    end
    
    max_str = sprintf('Max: %d Hz.\n',fVals(max_idx));
    
    hold on;
    plot(fVals(max_idx), Px(max_idx), 'r*');
    hold on;
    %text(fVals(max_idx)+4, Px(max_idx)-5, max_str);
    title(max_str, 'FontSize', 12)
    if plot_no==17
        xlabel('Frequency (Hz)')	 	 
    	ylabel(ylabel_name);
    end
    plot_no = plot_no + 1;
    
    
    
    fprintf('----------------------------------------------------------\n');
    fprintf('%s - slice %i\n', curr_tiff, slice_idx);
    fprintf('----------------------------------------------------------\n');
    %[sortedX,sortingIndices] = sort(Px(1:NFFT/2),'descend');
    [sortedX,sortingIndices] = sort(Px,'descend');
    for i=1:10
        fprintf('Rank %i power occurs at %d Hz.\n', i, fVals(sortingIndices(i))); 
    end



end
suptitle(curr_tiff)
figname = strcat(source_path, strcat(figY, sprintf('_run%i_slice%i.png', run_idx, slice_idx)));
% set(gcf,'units','normalized','outerposition',[0 0 1 1])
set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig, 'PaperPositionMode', 'auto')
saveas(fig, char(figname));

% print -dtif -r 150 test.tiff
% print(gcf, char(figname));

[sortedX,sortingIndices] = sort(Px(1:NFFT/2),'descend');
for i=1:10
    fprintf('Rank %i power occurs at %d Hz.\n', i, fVals(sortingIndices(i))); 
end


%%


figure();
plot(freqs, mag);
title('Double Sided FFT - with FFTShift');	 	 
xlabel('Frequency (Hz)')	 	 
ylabel('|DFT Values|');

L = length(signal);
Px=ft.*conj(ft)/(NFFT*L); %Power of each freq components	 	 
fVals=Fs*(-NFFT/2:NFFT/2-1)/NFFT;
figure();
plot(fVals,Px,'b');	 	 
title('Power Spectral Density');	 	 
xlabel('Frequency (Hz)')	 	 
ylabel('Power');


fVals=Fs*(0:NFFT/2-1)/NFFT;	 	 
plot(fVals,Px(1:NFFT/2),'b','LineSmoothing','on','LineWidth',1);	 	 
title('One Sided Power Spectral Density');	 	 
xlabel('Frequency (Hz)')	 	 
ylabel('PSD');
hold on;
% 
% [max_val,max_idx] = max(Px);
% fprintf('Maximum occurs at %d Hz.\n',fVals(max_idx));
% max_str = sprintf('Max at %d Hz.\n',fVals(max_idx));
% plot(fVals(max_idx), Px(max_idx), 'r*');
% hold on;
% text(fVals(max_idx)+4, Px(max_idx)-5, max_str);

% [sortedX,sortingIndices] = sort(Px(1:NFFT/2),'descend');
% for i=1:10
%    if fix(fVals(sortingIndices(i)))/100000 < 1
%        fprintf('Rank %i power occurs at %d kHz.\n', i, fVals(sortingIndices(i))/10000); 
%    else
%     fprintf('Rank %i power occurs at %d MHz.\n', i, fVals(sortingIndices(i))/1000000); 
%    end
% end


    
psdest = psd(spectrum.periodogram,signal,'Fs',Fs,'NFFT',length(signal));
[~,I] = max(psdest.Data);
fprintf('Maximum occurs at %d Hz.\n',psdest.Frequencies(I));


    
target_freq = 0.37;
freq_idx = find(abs((freqs-target_freq))==min(abs(freqs-target_freq)));

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




%% Get reference images
slice_dir = strcat(source_dir, sprintf('run%i_slices/', run_idx));
tseries_fn = curr_tiff;
avgimg_fn = sprintf('AVG/AVG_%s', curr_tiff);
ccimg_fn = strcat('CC/', sprintf('cc_%s.png', curr_tiff(1:end-4))); % , ridx, fidx));
stdimg_fn = sprintf('STD_%s', curr_tiff); %fov3_gratings_0000%i.tif #2.tif #%i.tif', ridx, fidx);

% Define field name for processed struct:
slice_no = ['slice' num2str(slice_idx)];
file_no = ['file' num2str(run_idx)]; % 'file1R'

struct_fn = strcat(source_dir, slice_no, '_traces_CC.mat')


%% Load reference images:

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
% masks=ROIselect_circle(ccimg);
masks=ROIselect_circle(avgimg);

%% save to struct:

D = struct;
D.(slice_no) = struct();
D.(slice_no).(file_no) = struct();

D.(slice_no).(file_no).masks = masks;
D.(slice_no).(file_no).avgimg = avgimg;
D.(slice_no).(file_no).ccimg = ccimg;
D.(slice_no).(file_no).tseries_fn = tseries_fn;
D.(slice_no).(file_no).slice_path = slice_dir;

    