
% clc; clear all;
% 
% %% Get Path to image
% 
%  % Set info manually:
% source = '/nas/volume1/2photon/projects';
% experiment = 'gratings_phaseMod';
% session = '20171009_CE059';
% acquisition = 'FOV1_zoom3x'; %'FOV1_zoom3x';
% tiff_source = 'functional'; %'functional_subset';
% acquisition_base_dir = fullfile(source, experiment, session, acquisition);
% curr_tiff_dir = fullfile(acquisition_base_dir, tiff_source);
% 
%this info is probably specific to this script
mcparams_id = I.mc_id; %sprintf('mcparams%02d',1);
roi_folder = I.roi_id; %'manual2D_poly';%name folder into which to save masks
roi_type = I.roi_method; %'polygon';%type of ROI to create ('circle','square',or 'polygon')
roi_slices = I.slices; %[1];%list of slices to create ROIs for
roi_ch = I.signal_channel; %1;%channel to use for ROI creation 

%----------USER INPUT STOPS HERE-----------
data_dir = A.data_dir.(I.analysis_id); %fullfile(acquisition_base_dir, tiff_source, 'DATA');

%load mcparams
mcparams = load(fullfile(data_dir,'mcparams.mat'));

%retrieve relevant mcparams sub-structure
mcparams = mcparams.(mcparams_id);

%use mcparams data to get paths to image
roi_file = mcparams.ref_file;%use ref file for motion correction to create ROIs

ch_path = fullfile(mcparams.source_dir, sprintf('Averaged_Slices_%s', mcparams.dest_dir), sprintf('Channel%02d', roi_ch), sprintf('File%03d', roi_file));

%create empty structure and cells to save roi info
roiparams = struct;
nrois = {};
sourcepaths = {};
maskpaths = {};


%% Go through slices and load images
for slidx = length(roi_slices);
    sl=roi_slices(slidx)
    %define slice name
    slicename = sprintf('average_Slice%02d_Channel%02d_File%03d.tif', sl, roi_ch, roi_file);

    %load image as double
    calcimg = tiffRead(fullfile(ch_path,slicename));

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
%     if strcmpi(roi_type,'circle')
%         [masks, RGBimg]=ROIselect_circle(calcimg_norm);
%     elseif strcmpi(roi_type,'square')
%         [masks, RGBimg]=ROIselect_square(calcimg_norm);
%     elseif strcmpi(roi_type,'polygon')
%         [masks, RGBimg]=ROIselect_polygon(calcimg_norm);
%     end
% 
    %% Save masks

    %defining target directories and filepaths
    roi_base_dir = fullfile(acquisition_base_dir,'ROIs',roi_folder);

    mask_dir = fullfile(roi_base_dir,'masks');
    if ~isdir(mask_dir)
        mkdir(mask_dir)
    end

    fig_dir = fullfile(roi_base_dir,'figures');
    if ~isdir(fig_dir)
        mkdir(fig_dir)
    end

    %save masks
    mask_filename = sprintf('%s_%s_Slice%02d_Channel%02d_masks.mat', session, acquisition, sl, roi_ch);
    
    %* if file exists, ask user to confirm before overwriting
    if exist(fullfile(mask_dir,mask_filename),'file')==2
        answer = inputdlg('File with masks already exists, overwite? (Y/N)');

        while ~ (strcmpi(answer,'Y') || strcmpi(answer,'N'))
            answer = inputdlg('File with masks already exists, overwite? (Y/N)');
        end

        if strcmpi(answer,'Y')%replace file
            save(fullfile(mask_dir,mask_filename),'masks');
        elseif strcmpi(answer,'N')%write file with extra id appended
            mask_filename = make_duplicate(mask_filename,mask_dir);
            save(fullfile(mask_dir,mask_filename),'masks');
        end
    else
        save(fullfile(mask_dir,mask_filename),'masks');
    end


    %save images
    fig_filename = sprintf('%s_%s_Slice%02d_Channel%02d_File%03d_masks.tif', session, acquisition, sl, roi_ch, roi_file);
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
    sourcepaths{slidx} = fullfile(ch_path,slicename);
    maskpaths{slidx} = fullfile(mask_dir,mask_filename);
    
end

%save roiparams
roiparams.nrois = nrois;
roiparams.sourceslices = roi_slices;
roiparams.sourcepaths = sourcepaths;
roiparams.maskpaths = maskpaths;

roiparams_path = fullfile(acquisition_base_dir,'ROIs', roi_folder, 'roiparams.mat');
save(roiparams_path,'-struct','roiparams');
