clear all;
clc;

%% DEFINE SOURCE DIRECTORY:

% Define source dir for current acquisition/experiment:
source_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/retinotopy1/';
struct_dir = fullfile(source_dir, 'datastructs');
if ~exist(struct_dir, 'dir')
    mkdir(struct_dir);
end

% Define TIFF dir if is sub-dir within the source-dir:
channel_idx = 1;
nchannels = 1;
%tiff_dir = sprintf('Corrected_Channel%02d', channel_idx);
tiff_dir = sprintf('Corrected');
acquisition_name = 'fov1_bar037Hz_run4';

%% Get SI volume info:

metaInfo = 'SI';

switch metaInfo
    case 'SI'

        metastruct_fn = sprintf('%s.mat', acquisition_name);
        meta = load(fullfile(source_dir, metastruct_fn));
        
        nvolumes = meta.(acquisition_name).metaDataSI.SI.hFastZ.numVolumes;
        nslices = meta.(acquisition_name).metaDataSI.SI.hFastZ.numFramesPerVolume;
        ndiscard = meta.(acquisition_name).metaDataSI.SI.hFastZ.numDiscardFlybackFrames;
        nframes_per_volume = nslices + ndiscard;
        ntotal_frames = nframes_per_volume * nvolumes;
        
        si_frame_times = meta.(acquisition_name).metaDataSI.frameTimestamps_sec(1:2:end);
        si_frame_rate = meta.(acquisition_name).metaDataSI.SI.hRoiManager.scanFrameRate;
        si_volume_rate = meta.(acquisition_name).metaDataSI.SI.hRoiManager.scanVolumeRate;
        
        frame_width = meta.(acquisition_name).metaDataSI.SI.hRoiManager.pixelsPerLine;
        slow_multiplier = meta.(acquisition_name).metaDataSI.SI.hRoiManager.scanAngleMultiplierSlow;
        lines_per_frame = meta.(acquisition_name).metaDataSI.SI.hRoiManager.linesPerFrame;
        frame_height = lines_per_frame/slow_multiplier;
        
        %clear M;
        
    case 'manual' % No motion-correction/processing, just using raw TIFFs.
        nvolumes = 350;
        nslices = 20;
        ndiscard = 0;
        
        nframes_per_volume = nslices + ndiscard;
        ntotal_frames = nframes_per_volume * nvolumes;
end

which_tiffs = sprintf('*Channel%02d*.tif', channel_idx);
tmp_tiffs = dir(fullfile(source_dir, tiff_dir, which_tiffs));
tiff_fns = {tmp_tiffs(:).name}';
ntiffs = length(tiff_fns) / nframes_per_volume;
fprintf('Found %i TIFF files for current acquisition analysis.\n', ntiffs);

%% Specify experiment parameters:

mw_path = fullfile(source_dir, 'mw_data');
mw = get_mw_info(mw_path, ntiffs);
pymat = mw.pymat;

% Create arbitrary stimtype codes:
stim_types = cell(1,length(mw.cond_types));
for sidx=1:length(mw.cond_types)
    sname = sprintf('code%i', sidx);
    stim_types{sidx} = sname;
end

% Get indices of each run to preserve order when indexing into MW
% file-structs:
for run_idx=1:length(mw.run_names)
    run_order.(mw.run_names{run_idx}) = pymat.(mw.run_names{run_idx}).ordernum + 1;
end

%% Grab traces for each slice and save a .mat for each run:

for tiff_idx=1:ntiffs
    
    curr_mw_fidx = mw.mw_fidx + tiff_idx - 1;
    
    % Make sure to grab the correct run based on TIFF order number:
    for order_no=1:length(fieldnames(run_order))
        if run_order.(mw.run_names{order_no}) == curr_mw_fidx
            curr_run_name = mw.run_names{order_no};
        end
    end
    
    mw_sec = (double(pymat.(curr_run_name).time) - double(pymat.(curr_run_name).time(1))) / 1E6;
    cycle_starts = pymat.(curr_run_name).idxs + 1; % Get indices of cycle starts
    mw_dur = (double(pymat.triggers(curr_mw_fidx,2)) - double(pymat.triggers(curr_mw_fidx,1))) / 1E6;
    
    if exist('si_frame_times')
        si_sec_vols = si_frame_times;
    else
        if isfield(pymat, 'ard_file_durs')
            ard_dur = double(pymat.ard_file_durs(curr_mw_fidx));
            %if sample_us==1 % Each frame has a t-stamped frame-onset (only true if ARD sampling every 200us, isntead of standard 1ms)
            si_sec_vols = (double(pymat.frame_onset_times{curr_mw_fidx}) - double(pymat.frame_onset_times{curr_mw_fidx}(1))) / 1E6; 
            if length(si_sec) < ntotal_frames % there are missed frame triggers
                si_sec_vols = linspace(0, ard_dur, ntotal_frames);
                % This is pretty accurate.. only off by ~ 3ms compared to SI's
                % trigger times.
            end
        else
            si_sec_vols = linspace(0, mw_dur, ntotal_frames);
        end
    end
    
    
    if pymat.stimtype=='bar'
        trim_long = 1;
        ncycles = pymat.info.ncycles;
        target_freq = pymat.info.target_freq;
        %si_frame_rate = 1/median(diff(si_sec_vols));
        %si_volume_rate = round(si_frame_rate/nframes_per_volume, 2); % 5.58%4.11 %4.26 %5.58
        n_true_frames = ceil((1/target_freq)*ncycles*si_volume_rate);
        
    end
    
    tiff.acquisition_name = acquisition_name;
    tiff.mw_run_name = curr_run_name;
    tiff.mw_fidx = curr_mw_fidx;
    tiff.mw_path = mw.mw_path;
    %tiff.run_fn = M.run_fns;
    tiff.mw_sec = mw_sec;
    tiff.stim_starts = cycle_starts;
    tiff.mw_dur = mw_dur;
    tiff.si_sec_vols = si_sec_vols;
    tiff.ncycles = ncycles;
    tiff.target_freq = target_freq;
    tiff.si_frame_rate = si_frame_rate;
    tiff.si_volume_rate = si_volume_rate;
    tiff.n_true_frames = n_true_frames;
    
    tiff.nvolumes = nvolumes;
    tiff.nslices = nslices;
    tiff.ntotal_frames = ntotal_frames;
    tiff.nframes_per_volume = nframes_per_volume;
    tiff.tiff_path = fullfile(source_dir, tiff_dir);
    tiff.imgX = frame_width;
    tiff.imgY = frame_height;
   
    curr_struct_name = char(sprintf('meta_%05d.mat', tiff_idx));
    save(fullfile(struct_dir, curr_struct_name), '-struct', 'tiff');
    
end
    
%% Get raw traces from TIFFs:

% =========================================================================
% Parameters:
% =========================================================================
corrected = 'acquisition2p';
% -----------------------
roi_type = 'create_rois';
% roi_type: 
%   'create_rois' : create new ROIs using circle-GUI
%   'smoothed_pixels' : use all pixels, but smooth with kernel size,
%   ksize=2 (default)
%   'prev_rois' : use previosuly-created ROIs 
% ----------
% TO DO:  1. option to specify kernel size
%         2.  let user select which ROI set to use in a non-stupid way...

channel_idx = 1;

tiff_info_structs = dir(fullfile(struct_dir, '*meta_*'));
tiff_info_structs = {tiff_info_structs(:).name}';
tiff_info = load(fullfile(struct_dir, tiff_info_structs{1}));

% =========================================================================
% Grab traces, and save them:
% =========================================================================
% TODO:  1. don't have separate "traces" structs from the meta struct --
% combine into "acquisition-struct" for easier analysis??

raw_traces = get_raw_traces(corrected, acquisition_name, ntiffs, nchannels, tiff_info, roi_type, channel_idx);

% raw_traces.slice(10).traces.file(1)
% currently, have it s.t. new ROI masks are created for each slice, but
% re-used for each file (of that slice)
% -------------------------------------------------------------------------

curr_tracestruct_name = char(sprintf('traces_%05d.mat', 1));
save(fullfile(struct_dir, curr_tracestruct_name), '-struct', 'raw_traces');

%

%%

scalevec = [2 1 1];
tmp_source_dir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/retinotopy037Hz';
tmp_tiff_dir = 'Corrected_Channel01_File003';
resize_dir = fullfile(tmp_source_dir, strcat(tmp_tiff_dir, '_scaled'));
if ~exist(resize_dir, 'dir')
    mkdir(resize_dir)
end
tmp_tiffs = dir(fullfile(tmp_source_dir, tmp_tiff_dir, '*.tif'));
tmp_tiffs = {tmp_tiffs(:).name}';

for tmp_idx=1:length(tmp_tiffs)
    %tmp_tiff = sprintf('%s_Slice%02d_Channel01_File003.tif', acquisition_name, tmp_idx);
    tmp_tiff = tmp_tiffs{tmp_idx};
    tmp_tiff_path = fullfile(tmp_source_dir, tmp_tiff_dir, tmp_tiff);
    imData=bigread2_scale(tmp_tiff_path,1,[],scalevec);
    
    resize_dir = fullfile(tmp_source_dir, strcat(tmp_tiff_dir, '_scaled'));
    tiffWrite(imData, tmp_tiff, resize_dir);
end

