get_rois_manual2D

clc; clear all;
% 
% %% Get Path to image
%  % Set info manually:
rootdir = '/nas/volume1/2photon/data';
animalid = 'JR063';
session = '20171128_JR063_testbig';
acquisition = 'FOV1_zoom1x'; %'FOV1_zoom3x';
run = 'gratings_static_run1';

tiffsource = 'processed002';
sourcetype = 'mcorrected';

roi_slices = '';
%roi_type = 'manual2D_circle';
roi_name = 'rois002';
roi_hashid = '828a01';

roi_dir = fullfile(rootdir, animalid, session, 'ROIs');
acquisition_dir = fullfile(rootdir, animalid, session, acquisition);
runinfo = loadjson(fullfile(acquisition_dir, run, sprintf('%s.json', run)));
pidinfo = loadjson(fullfile(acquisition_dir, run, 'processed', sprintf('pids_%s.json', run)));
roiinfo = loadjson(fullfile(roi_dir, sprintf('rids_%s.json', session)));

if ismember(roi_name, fieldnames(roiinfo))
    roikey = roi_name;
else
    roikey = sprintf('%s_%s', roi_name, roi_hashid);
end
roiparams = roiinfo.(roikey)

roi_type = roiparams.roi_type

% if any(strfind(tiffsource, 'processed'))
%     tiffsource_parent = fullfile(acquisition_dir, run, 'processed');
%     tiff_opts = dir(fullfile(tiffsource_parent, [tiffsource '*']));
% else
%     tiffsource_parent = fullfile(acquisition_dir, run);
%     tiff_opts = dir(fullfile(tiffsource_parent, 'raw*'));
% end
% tiffsource_id = tiff_opts.name;
% tiffsource_path = fullfile(tiffsource_parent, tiffsource_id);
% 
% source_opts = dir(fullfile(tiffsource_path, [sourcetype '*']));
% if length(source_opts) == 0
%     fprintf('Unable to find sourcetype specified -%s- in %s.\n', sourcetype, tiffsource_path);
%     fprintf('Found the following alt sources:\n');
%     alts = dir(tiffsource_path);
%     for a=1:length(alts)
%         display(alts(a).name);
%     end
% elseif length(source_opts) > 1
%     for s=1:length(source_opts)
%         if length(strfind(source_opts(s).name, '_')) == 1
%             sourcetype_dir = source_opts(s).name;
%         end
%     end
% else
%     sourcetype_dir = source_opts(1).name;
% end

% Get AVERAGE SLICE files:
% average_images_dir = fullfile(tiffsource_path, [sourcetype_dir '_average_slices']);
average_images_dir = [roiparams.SRC '_average_slices'];
average_image_fns = dir(fullfile(average_images_dir, '*.tif'));
average_image_fns = {average_image_fns(:).name}';

[src_parent, src_name, ~] = fileparts(roiparams.SRC);
[src_root, tiffsource, ~] = fileparts(src_parent);
if length(strfind(tiffsource, '_'))==1
    tiffsrc_pts = strsplit(tiffsource, '_');
    tiffsource_name = tiffsrc_pts{1};
    tiffsource_id = tiffsrc_pts{2};
else
    tiffsource_name = tiffsource;
    tiffsource_id = '';
end

% Get REFERENCE info:
if any(strfind(src_name, 'mcorrected'))
    % Motion-corrected, so use whatever reference file was used as ref
    if ismember(tiffsource, fieldnames(pidinfo))
        pidkey = tiffsource;
    else
        pidkey = tiffsource_name;
    end
    ref_file = pidinfo.(pidkey).PARAMS.motion.ref_file;
    ref_channel = pidinfo.(pidkey).PARAMS.motion.ref_channel;
else
    ref_file = 1;
    ref_channel = 1;
end

if length(roi_slices) == 0
    roi_slices = runinfo.slices;
end

%create empty structure and cells to save roi info
roiparams = struct;
nrois = {};
sourcepaths = {};
maskpaths = {};


%% Go through slices and load images
for slidx = length(roi_slices)
    curr_slice = roi_slices(slidx)
    
    % define slice name
    curr_slice_fn = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', acquisition, curr_slice, ref_channel, ref_file)

    %load image as double
    calcimg = tiffRead(fullfile(average_images_dir, curr_slice_fn));

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
    roi_base_dir = roiparams.DST;

    mask_dir = fullfile(roi_base_dir,'masks');
    if ~isdir(mask_dir)
        mkdir(mask_dir)
    end

    fig_dir = fullfile(roi_base_dir,'figures');
    if ~isdir(fig_dir)
        mkdir(fig_dir)
    end

    %save masks
    mask_filename = sprintf('%s_%s_Slice%02d_Channel%02d_masks.mat', session, acquisition, curr_slice, ref_channel);
    
    %* if file exists, ask user to confirm before overwriting
    if exist(fullfile(mask_dir,mask_filename),'file')==2
        answer = inputdlg('File with masks already exists, overwite? (Y/N)');

        while ~ (strcmpi(answer,'Y') || strcmpi(answer,'N'))
            answer = inputdlg('File with masks already exists, overwite? (Y/N)');
        end

        if strcmpi(answer,'Y')%replace file
            save(fullfile(mask_dir,mask_filename),'masks');
        elseif stparcmpi(answer,'N')%write file with extra id appended
            mask_filename = make_duplicate(mask_filename,mask_dir);
            save(fullfile(mask_dir,mask_filename),'masks');
        end
    else
        save(fullfile(mask_dir,mask_filename),'masks');
    end


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
    maskpaths{slidx} = fullfile(mask_dir,mask_filename);
    
end

%save roiparams
ROIPARAMS.nrois = nrois;
ROIPARAMS.sourceslices = roi_slices;
ROIPARAMS.sourcepaths = sourcepaths;
ROIPARAMS.maskpaths = maskpaths;

roiparams_path = fullfile(roi_base_dir, 'roiparams.mat');
save(roiparams_path,'-struct','ROIPARAMS');

fprintf('Saved ROIs to: %s\n', roi_base_dir);
