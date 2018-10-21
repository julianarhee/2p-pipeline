
clc; clear all;
% 
%% Set info manually:
% rootdir = '/nas/volume1/2photon/data';
% animalid = 'JR063'; %'JR063';
% session = '20171128_JR063'; %'20171202_JR063';
% roi_id = 'rois007'; %'e4893c';

rootdir = '/n/coxfs01/2p-data' %'/mnt/odyssey'
animalid = 'JC022' %'JR063';
session = '20181016' %'20171202_JR063';
roi_id = 'rois001' %'e4893c';

%% Load RID parameter set:
roi_dir = fullfile(rootdir, animalid, session, 'ROIs');
roidict = loadjson(fullfile(roi_dir, sprintf('rids_%s.json', session)));
roi_ids = fieldnames(roidict);
assert(ismember(roi_id, roi_ids), 'ROI ID %s not found in roidict entries at\n%s', roi_id, roi_dir);
RID = roidict.(roi_id)

roi_type = RID.roi_type;
zproj_type = RID.PARAMS.options.zproj_type
roi_hash = RID.rid_hash;
roi_slices = RID.PARAMS.options.slices

%% Get paths:
%animalroot = strsplit(RID.SRC, session);
animalroot = strsplit(RID.PARAMS.tiff_sourcedir, session);
sessionparts = strsplit(animalroot{2}, '/');
acquisition = sessionparts{2};
run = sessionparts{3};
if length(sessionparts(4:end)) > 2
    is_processed = true;
    tiffsource = sessionparts{5};
    tiffsource_pts = strsplit(tiffsource, '_');
    tiffsource_id = tiffsource_pts{1};
    source_type = sessionparts{6};
    post_rundir_path = fullfile('processed', tiffsource, source_type);
else
    is_processed = false;
    tiffsource = sessionparts{4};
    tiffsource_pts = strsplit(tiffsource, '_');
    tiffsource_id = tiffsource_pts{1};
    source_type = '';
    post_rundir_path = tiffsource;
end

%% Get reference file/channel for manual ROI selection:
acquisition_dir = fullfile(rootdir, animalid, session, acquisition);
runinfo = loadjson(fullfile(acquisition_dir, run, sprintf('%s.json', run)));
if is_processed
    pidinfo = loadjson(fullfile(acquisition_dir, run, 'processed', sprintf('pids_%s.json', run)));
    if pidinfo.(tiffsource_id).PARAMS.motion.correct_motion
        mc_ref_file = pidinfo.(tiffsource_id).PARAMS.motion.ref_file;
        mc_ref_channel = pidinfo.(tiffsource_id).PARAMS.motion.ref_channel;
    end
else
    mc_ref_file = 1;
    mc_ref_channel = 1;
end
rid_ref_file = RID.PARAMS.options.ref_file;
rid_ref_channel = RID.PARAMS.options.ref_channel;

if rid_ref_file~=mc_ref_file
    fprintf('*****WARNING*****')
    fprintf('Specified RID reference file does not match MC reference.\n')
    fprintf('Using ref file %i for ROI extraction (MC: ref file is %i)\n', rid_ref_file, mc_ref_file)
end
if rid_ref_channel~=mc_ref_channel
    fprintf('*****WARNING*****')
    fprintf('Specified RID reference channel does not match MC reference.\n')
    fprintf('Using ref channel %i for ROI extraction (MC: ref channel is %i)\n', rid_ref_channel, mc_ref_channel)
end

ref_file = rid_ref_file;
ref_channel = rid_ref_channel;

ref_filename = sprintf('File%03d', ref_file);
ref_channelname = sprintf('Channel%02d', ref_channel);
fprintf('RID %s -- Using %s, %s as reference for manual ROI creation.\n', roi_hash, ref_filename, ref_channelname);


if length(roi_slices)==0 || ~ismember('slices', fieldnames(RID.PARAMS.options))
    roi_slices = runinfo.slices;
    fprintf('RID %s -- Slices not specified for manual ROI creation. Using all %i slices.\n', roi_hash, length(runinfo.slices));
else
    fprintf('RID %s -- Specified slices %s.\n', roi_hash, mat2str(roi_slices));
end
    
if length(runinfo.slices) > 1
    is_3D = true
else
    is_3D = false
end

%% Get AVERAGE SLICE files:
% average_images_dir = fullfile(tiffsource_path, [sourcetype_dir '_average_slices']);
rid_src_dir = RID.SRC;
if ~any(strfind(rid_src_dir, rootdir))
    orig_root = rid_src_dir(1:strfind(rid_src_dir, sprintf('/%s/%s', animalid, session))-1);
    fprintf('Replacing root dir...\n');
    rid_src_dir = strrep(rid_src_dir, orig_root, rootdir);
end

if strfind(zproj_type, '_warped_')
    [slice_sourcedir, slice_tiff_fn, ext] = fileparts(rid_src_dir);
    slice_tiffs = {[slice_tiff_fn ext]};
else
    average_images_dir = [rid_src_dir sprintf('_%s_deinterleaved', zproj_type)];
    if length(dir(fullfile(average_images_dir, '*.tif'))) == 0
        slice_sourcedir = fullfile(average_images_dir, ref_channelname, ref_filename);
    else
        slice_sourcedir = average_images_dir;
    end
    slice_tiffs = dir(fullfile(slice_sourcedir, '*.tif'));
    slice_tiffs = {slice_tiffs(:).name}';
end

assert(length(slice_tiffs)==length(roi_slices), 'RID %s -- WARNING:  Incorrect number of slices found in specified dir:\n%s\n', roi_hash, slice_sourcedir);


%% create empty structure and cells to save roi info
roiparams = struct;
nrois = {};

sourcepaths = {};
% maskpaths = {};
allmasks = {};
zproj_imgs = {};

% Go through slices and load images
for slidx = roi_slices
    if max(size(roi_slices)) == 1
        curr_slice = roi_slices;
    else
        curr_slice = roi_slices(slidx)
    end
    
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
    
    %% Check that there are no empty masks:
    non_empty = arrayfun(@(i) length(find(masks(:,:,i))) > 0, 1:size(masks,3));
    masks = masks(:,:,non_empty);

    %% Save masks

    %defining target directories and filepaths
    roi_base_dir = RID.DST;
    if ~any(strfind(roi_base_dir, rootdir))
        orig_root = roi_base_dir(1:strfind(roi_base_dir, sprintf('/%s/%s', animalid, session))-1);
        fprintf('Replacing root dir...\n');
        roi_base_dir = strrep(roi_base_dir, orig_root, rootdir);
    end

%     mask_dir = fullfile(roi_base_dir,'masks');
%     if ~isdir(mask_dir)
%         mkdir(mask_dir)
%     end

    fig_dir = fullfile(roi_base_dir,'figures');
    if ~isdir(fig_dir)
        mkdir(fig_dir)
    end

    %save images
    %fig_filename = sprintf('%s_%s_Slice%02d_Channel%02d_File%03d_masks.tif', session, acquisition, curr_slice, ref_channel, ref_file);
       fig_filename = sprintf('%s_%s_Slice%02d_Channel%02d_File%03d_masks.tif', roi_id, roi_hash, curr_slice, ref_channel, ref_file);
    %* check if file exists
    if exist(fullfile(fig_dir,fig_filename),'file')==2
        answer = input('Found existing mask figure. Overwrite? Press <Y> or <N>: ', 's');
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
     sourcepaths{slidx} = fullfile(slice_sourcedir, curr_slice_fn); %fullfile(average_images_dir, curr_slice_fn);
%     maskpaths{slidx} = fullfile(mask_dir,mask_filename);
     allmasks{slidx} = masks;

     zproj_imgs{slidx} = calcimg;
     

end
% masks = zeros(512, 512, 65);
% for r = 1:65
%     mask = hdf5read(mfilepath, sprintf('/masks/slice1/roi%i', r));
%     masks(:,:,r) = mask;
% end
% RGBimg = zeros([size(calcimg),3]);
% RGBimg(:,:,1)=0;
% RGBimg(:,:,2)=calcimg_norm;
% RGBimg(:,:,3)=0;
% for r = 1:65
%     masks_ones=find(squeeze(masks(:,:,r))==1);
%     RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,r);
%     RGBimg(:,:,1)=RGBimg(:,:,1)+0.5*masks(:,:,r);
% end
% figure(); imshow(RGBimg);

% Save ROI masks by slice:
%maskname = sprintf('masks_%s', roi_hash)
%mask_fn = fullfile(mask_dir, sprintf('%s.h5', maskname));
%mask_fn = fullfile(roi_base_dir, sprintf('%s.hdf5', maskname));

curr_filename = sprintf('File%03d', ref_file);

mask_fn = fullfile(roi_base_dir, 'masks.hdf5');
% 
% fcpl = H5P.create('H5P_FILE_CREATE');
% fapl = H5P.create('H5P_FILE_ACCESS');
% fid = H5F.create(mask_fn,'H5F_ACC_TRUNC',fcpl,fapl);

%plist = 'H5P_DEFAULT';
%fgroup = H5G.create(mask_fn, sprint('/%s', curr_filename), plist, plist, plist);
%mgroup = H5G.create(fgroup, 'masks', plist, plist, plist);
%zgroup = H5G.create(fgroup, 'zproj_img', plist, plist, plist);
for slidx = 1:length(sourcepaths)
    masks = allmasks{slidx};
    curr_slice = roi_slices(slidx);
    h5create(mask_fn, sprintf('/%s/masks/Slice%02d', curr_filename, curr_slice), size(masks))
    h5write(mask_fn, sprintf('/%s/masks/Slice%02d', curr_filename, curr_slice), masks)
    h5writeatt(mask_fn,  sprintf('/%s/masks/Slice%02d', curr_filename, curr_slice) ,'source_file', sourcepaths{slidx})
    h5writeatt(mask_fn, sprintf('/%s/masks/Slice%02d', curr_filename, curr_slice) ,'nrois', nrois{slidx})
    h5writeatt(mask_fn, sprintf('/%s/masks/Slice%02d', curr_filename, curr_slice) ,'src_roi_idxs', [1:nrois{slidx}])
    
    h5create(mask_fn, sprintf('/%s/zproj_img/Slice%02d', curr_filename, curr_slice), size(zproj_imgs{slidx}))
    h5write(mask_fn, sprintf('/%s/zproj_img/Slice%02d', curr_filename, curr_slice), zproj_imgs{slidx})
    h5writeatt(mask_fn,  sprintf('/%s/zproj_img/Slice%02d', curr_filename, curr_slice) ,'source_file', sourcepaths{slidx})
end
h5disp(mask_fn)
h5writeatt(mask_fn, sprintf('/%s/masks', curr_filename), 'source', slice_sourcedir)
%h5writeatt(mask_fn,  '/', 'source', slice_sourcedir);

h5writeatt(mask_fn,  '/', 'creation_date', datestr(now));
h5writeatt(mask_fn,  '/', 'roi_type', roi_type);
h5writeatt(mask_fn,  '/', 'roi_id', roi_id);
h5writeatt(mask_fn,  '/', 'roi_hash', roi_hash);
h5writeatt(mask_fn,  '/', 'animal', animalid);
h5writeatt(mask_fn,  '/', 'session', session);
h5writeatt(mask_fn,  '/', 'ref_file', ref_file);
h5writeatt(mask_fn,  '/', 'creation_date', datestr(now));

h5writeatt(mask_fn,  '/', 'keep_good_rois', 0);
h5writeatt(mask_fn,  '/', 'ntiffs_in_set', 1); % shoudl this be n actual tiffs? 1 since only using a ref file
h5writeatt(mask_fn,  '/', 'zproj', zproj_type);



%h5writeatt(mask_fn,  sprintf('/masks'), 'acquisition', acquisition);
%h5writeatt(mask_fn,  sprintf('/masks'), 'run', run);
%h5writeatt(mask_fn,  sprintf('/masks'), 'zproj', zproj_type);
%h5writeatt(mask_fn,  '/', 'ref_file', ref_file);
%h5writeatt(mask_fn,  sprintf('/masks'), 'ref_channel', ref_channel);
if is_3D
    is3D_val = 'True';
else
    is3D_val = 'False';
end
h5writeatt(mask_fn,  '/', 'is_3D', is3D_val);


% h5disp(mask_fn)
info = hdf5info(mask_fn);

%
for a=1:length(info.GroupHierarchy.Groups(1).Attributes)
    roiset_attr = info.GroupHierarchy.Groups(1).Attributes;
    if isnumeric(roiset_attr(a).Value)
        fprintf('%s: %i\n', roiset_attr(a).Name, roiset_attr(a).Value);
    else
        fprintf('%s: %s\n', roiset_attr(a).Name, roiset_attr(a).Value.Data);
    end
end

% Clean up tmp file

tmp_rid_dir = fullfile(roi_dir, 'tmp_rids');
tmp_rid_fn = sprintf('tmp_rid_%s.json', roi_hash);

finished_rid_dir = fullfile(tmp_rid_dir, 'completed');
if ~exist(finished_rid_dir)
    mkdir(finished_rid_dir)
end

if exist(fullfile(tmp_rid_dir, tmp_rid_fn), 'file')
    movefile(fullfile(tmp_rid_dir, tmp_rid_fn), fullfile(finished_rid_dir, tmp_rid_fn));
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
