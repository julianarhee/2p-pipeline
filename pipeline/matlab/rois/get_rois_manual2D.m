get_rois_manual2D

clc; clear all;
% 
%% Set info manually:
rootdir = '/nas/volume1/2photon/data';
animalid = 'JR063';
session = '20171202_JR063';

roi_slices = '';

rid_hash = 'e4893c';
zproj_type = 'mean';

%% Load RID parameter set:
roi_dir = fullfile(rootdir, animalid, session, 'ROIs');
roidict = loadjson(fullfile(roi_dir, sprintf('rids_%s.json', session)));
roi_ids = fieldnames(roidict);
for rkey = 1:length(fieldnames(roidict))
    if strcmp(roidict.(roi_ids{rkey}).rid_hash, rid_hash)
        roi_id = roi_ids{rkey};
        RID = roidict.(roi_id);
    end   
end
roi_type = RID.roi_type;


%% Get paths:
animalroot = strsplit(RID.SRC, session);
sessionparts = strsplit(animalroot{2}, '/');
acquisition = sessionparts{2};
run = sessionparts{3};
if length(sessionparts(4:end)) > 2
    is_processed = true;
    tiffsource_name = sessionparts{5};
    tiffsource_pts = strsplit(tiffsource_name, '_');
    tiffsource = tiffsource_pts{1};
    sourcetype_name = sessionparts{6};
    post_rundir_path = fullfile('processed', tiffsource_name, sourcetype_name);
else
    is_processed = false;
    tiffsource_name = sessionparts{4};
    tiffsource_pts = strsplit(tiffsource_name, '_');
    tiffsource = tiffsource_pts{1};
    sourcetype_name = '';
    post_rundir_path = tiffsource_name;
end

%% Get reference file/channel for manual ROI selection:
acquisition_dir = fullfile(rootdir, animalid, session, acquisition);
runinfo = loadjson(fullfile(acquisition_dir, run, sprintf('%s.json', run)));
if is_processed
    pidinfo = loadjson(fullfile(acquisition_dir, run, 'processed', sprintf('pids_%s.json', run)));
    if pidinfo.(tiffsource).PARAMS.motion.correct_motion
        ref_file = pidinfo.(tiffsource).PARAMS.motion.ref_file;
        ref_channel = pidinfo.(tiffsource).PARAMS.motion.ref_channel;
    end
else
    ref_file = 1;
    ref_channel = 1;
end
ref_filename = sprintf('File%03d', ref_file);
ref_channelname = sprintf('Channel%02d', ref_channel);
fprintf('RID %s -- Using %s, %s as reference for manual ROI creation.\n', rid_hash, ref_filename, ref_channelname);


if length(roi_slices) == 0 || ~ismember('slices', fieldnames(RID))
    roi_slices = runinfo.slices;
    fprintf('RID %s -- Slices not specified for manual ROI creation. Using all %i slices.\n', rid_hash, length(runinfo.slices));
else
    fprintf('RID %s -- Specified slices %s.\n', rid_hash, mat2str(roi_slices));
end
    

%% Get AVERAGE SLICE files:
% average_images_dir = fullfile(tiffsource_path, [sourcetype_dir '_average_slices']);
average_images_dir = [RID.SRC sprintf('_%s_slices', zproj_type)];
if length(dir(fullfile(average_images_dir, '*.tif'))) == 0
    slice_sourcedir = fullfile(average_images_dir, ref_channelname, ref_filename);
else
    slice_sourcedir = average_images_dir;
end
slice_tiffs = dir(fullfile(slice_sourcedir, '*.tif'));
slice_tiffs = {slice_tiffs(:).name}';
if length(slice_tiffs) == length(roi_slices)
    fprintf('RID %s -- Found expected number of slice images in:\n%s\n', rid_hash, slice_sourcedir);
else
    fprintf('RID %s -- WARNING:  Incorrect number of slices found in specified dir:\n%s\n', rid_hash, slice_sourcedir);
end



%% create empty structure and cells to save roi info
roiparams = struct;
nrois = {};
sourcepaths = {};
% maskpaths = {};
allmasks = {};

% Go through slices and load images
for slidx = roi_slices
    curr_slice = roi_slices(slidx)
    
    % define slice name
    curr_slice_fn = slice_tiffs{curr_slice}; %sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition, curr_slice, ref_channel, ref_file)

    %load image as double
    calcimg = tiffRead(fullfile(slice_sourcedir, curr_slice_fn));

    %normalize image
    calcimg_norm = (calcimg-min(calcimg(:)))./(max(calcimg(:))-min(calcimg(:)));

    %% Get masks
    switch roi_type
        case 'manual2D_circle'
            [masks, RGBimg]=ROIselect_circle(calcimg_norm);
        case 'manual2D_square'
            [masks, RGBimg]=ROIselect_square(calcimg_norm);
        case 'manual2D_polygon'
            [masks, RGBimg]=ROIselect_polygon(calcimg_norm);
        otherwise
            % some default thing
    end

    %% Save masks

    %defining target directories and filepaths
    roi_base_dir = RID.DST;

    mask_dir = fullfile(roi_base_dir,'masks');
    if ~isdir(mask_dir)
        mkdir(mask_dir)
    end

    fig_dir = fullfile(roi_base_dir,'figures');
    if ~isdir(fig_dir)
        mkdir(fig_dir)
    end

    %save masks
    %mask_filename = sprintf('%s_%s_Slice%02d_Channel%02d_masks.mat', session, acquisition, curr_slice, ref_channel);
    %mask_filename = 'masks.h5';
    
%     %* if file exists, ask user to confirm before overwriting
%     if exist(fullfile(mask_dir,mask_filename),'file')==2
%         answer = inputdlg('File with masks already exists, overwite? (Y/N)');
% 
%         while ~ (strcmpi(answer,'Y') || strcmpi(answer,'N'))
%             answer = inputdlg('File with masks already exists, overwite? (Y/N)');
%         end
% 
%         if strcmpi(answer,'Y')%replace file
%             save(fullfile(mask_dir,mask_filename),'masks');
%         elseif stparcmpi(answer,'N')%write file with extra id appended
%             mask_filename = make_duplicate(mask_filename,mask_dir);
%             save(fullfile(mask_dir,mask_filename),'masks');
%         end
%     else
%         save(fullfile(mask_dir,mask_filename),'masks');
%     end


    %save images
    fig_filename = sprintf('%s_%s_Slice%02d_Channel%02d_File%03d_masks.tif', session, acquisition, curr_slice, ref_channel, ref_file);
    %* check if file exists
    if exist(fullfile(fig_dir,fig_filename),'file')==2
        if strcmpi(answer,'Y')%replace file
            imwrite(uint16(RGBimg*(2^16)),fullfile(fig_dir,fig_filename),'TIFF');
        elseif strcmpi(answer,'N')%write file with extra id appended
            mask_filename = make_duplicate(fig_filename,mask_dir);
            imwrite(uint16(RGBimg*(2^16)),fullfile(fig_dir,mask_filename),'TIFF');
        end
    else
        imwrite(uint16(RGBimg*(2^16)),fullfile(fig_dir,fig_filename),'TIFF');
    end
    
    %keep track of info to save to roiparams
    nrois{slidx} = size(masks,3);
    sourcepaths{slidx} = fullfile(average_images_dir, curr_slice_fn);
%     maskpaths{slidx} = fullfile(mask_dir,mask_filename);
    allmasks{slidx} = masks;


end



%% Save ROI masks by slice:
mask_fn = fullfile(mask_dir, 'masks.h5');

for slidx = 1:length(sourcepaths)
    masks = allmasks{slidx};
    curr_slice = roi_slices(slidx);
    for curr_roi = 1:size(masks,3)
        %h5create(mask_fn, sprintf('/masks/slice%i/roi%i', curr_slice, curr_roi), size(masks(:,:,curr_roi)))
        h5write(mask_fn, sprintf('/masks/slice%i/roi%i', curr_slice, curr_roi), masks(:,:,curr_roi))
    end
    h5writeatt(mask_fn,  sprintf('/masks/slice%i', curr_slice) ,'creation_date', datestr(now))
    h5writeatt(mask_fn,  sprintf('/masks/slice%i', curr_slice) ,'source', sourcepaths{slidx})
end
h5writeatt(mask_fn,  sprintf('/masks'), 'creation_date', datestr(now));
h5writeatt(mask_fn,  sprintf('/masks'), 'roi_type', roi_type);
h5writeatt(mask_fn,  sprintf('/masks'), 'roi_hash', rid_hash);
h5writeatt(mask_fn,  sprintf('/masks'), 'animal', animalid);
h5writeatt(mask_fn,  sprintf('/masks'), 'session', session);
h5writeatt(mask_fn,  sprintf('/masks'), 'acquisition', acquisition);
h5writeatt(mask_fn,  sprintf('/masks'), 'run', run);
h5writeatt(mask_fn,  sprintf('/masks'), 'zproj', zproj_type);

% h5disp(mask_fn)
info = hdf5info(mask_fn)

%
for a=1:length(info.GroupHierarchy.Groups(1).Attributes)
    roiset_attr = info.GroupHierarchy.Groups(1).Attributes;
    fprintf('%s: %s\n', roiset_attr(a).Name, roiset_attr(a).Value.Data);
end

    
%%
% 
% %save roiparams
% ROIPARAMS.nrois = nrois;
% ROIPARAMS.sourceslices = roi_slices;
% ROIPARAMS.sourcepaths = sourcepaths;
% ROIPARAMS.maskpaths = maskpaths;
% 
% roiparams_path = fullfile(roi_base_dir, 'roiparamsh5.mat');
% save(roiparams_path,'-struct','ROIPARAMS');
% 
% fprintf('Saved ROIs to: %s\n', roi_base_dir);


%%
