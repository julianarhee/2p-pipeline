

clear all

%% Set info manually:
source = '/nas/volume1/2photon/projects';
experiment = 'gratings_phaseMod';
session = '20171009_CE059';
acquisition = 'FOV1_zoom3x'; %'FOV1_zoom3x';
tiff_source = 'functional'; %'functional_subset';
acquisition_base_dir = fullfile(source, experiment, session, acquisition);

%%
aid = 1;
analysis_id = sprintf('analysis%02d',1);
roi_folder1 = 'blobs_DoG';%name of ROI subfolder



%% load reference file
path_to_reference = fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source));
A = load(path_to_reference);
if ~isfield(A,'roi_dir')
    A.roi_dir = fullfile(acquisition_base_dir,'ROIs');
end

%roi_folder = 'blobs_DoG';%name of ROI subfolder

%% load roiparams
roiparams_path = fullfile(A.roi_dir, roi_folder1, 'roiparams.mat');
roiparams = load(roiparams_path);

%create options structure
options = struct;
options.metrics_folder = fullfile(A.roi_dir, roi_folder1, 'metrics');
options.slices = [1];
options.ntiffs = A.ntiffs;
options.channel = A.signal_channel.(analysis_id);

 %% unpack ROI metrics-this could be turned into a fxn.....somehow
for sidx = 1:length(options.slices)
    sl = options.slices(sidx)

    %load pixel-to-pixel correlation
    filename = sprintf('Slice%02d_Channel%02d_pixcorrstruct.mat', sl, options.channel);
    pixcorrstruct = load(fullfile(options.metrics_folder, filename));

%         %load trial-to-trial correlation
%         filename = sprintf('Slice%02d_Channel%02d_filecorrstruct.mat', sl, options.channel);
%         filecorrstruct = load(fullfile(options.metrics_folder, filename));

    %load trial-to-trial correlation
    filename = sprintf('Slice%02d_Channel%02d_trialcorrstruct.mat', sl, options.channel);
    trialcorrstruct = load(fullfile(options.metrics_folder, filename));

    %put into list
    pixel_corr_all = zeros(options.ntiffs,length(pixcorrstruct.file(1).mean_pixel_corr));
    for fidx=1:options.ntiffs
        pixel_corr_all(fidx,:)=pixcorrstruct.file(fidx).mean_pixel_corr;
    end

    %keep track of which slice and roi values came from 
    if sidx==1
        pixel_corr_roi = nanmean(pixel_corr_all,1);%average across file
        trial_corr_roi = max(trialcorrstruct.mean_trial_corr,[],2)';%take the max across stims
        %  file_corr_roi = filecorrstruct.mean_file_corr;
        slice_src = ones(1,size(pixel_corr_all,2))*double(sl);
        roi_src = 1:size(pixel_corr_all,2);
    else
        pixel_corr_roi = [pixel_corr_roi nanmean(pixel_corr_all,1)];%average across file
        trial_corr_roi = [trial_corr_roi max(trialcorrstruct.mean_trial_corr,[],2)'];%take the max across stims
       % file_corr_roi = [file_corr_roi filecorrstruct.mean_file_corr];
        slice_src =[slice_src ones(1,size(pixel_corr_all,2))*double(sl)];
        roi_src = [roi_src 1:size(pixel_corr_all,2)];
    end
end
trial_corr_roi1 = trial_corr_roi;
pixel_corr_roi1 = pixel_corr_roi;
slice_src1 = slice_src;
roi_src1 = roi_src;


%% laod ROIs2

roi_folder2 = 'blobs_DoG2'; %manual2D';%name of ROI subfolder
analysis_id = 'analysis04';

%load roiparams
roiparams_path = fullfile(A.roi_dir, roi_folder2, 'roiparams.mat');
roiparams = load(roiparams_path);

%create options structure
options = struct;
options.metrics_folder = fullfile(A.roi_dir, roi_folder2, 'metrics');
options.slices = [1];
options.ntiffs = A.ntiffs;
options.channel = A.signal_channel.(analysis_id);

%unpack ROI metrics-this could be turned into a fxn.....somehow
for sidx = 1:length(options.slices)
    sl = options.slices(sidx)

    %load pixel-to-pixel correlation
    filename = sprintf('Slice%02d_Channel%02d_pixcorrstruct.mat', sl, options.channel);
    pixcorrstruct = load(fullfile(options.metrics_folder, filename));

%         %load trial-to-trial correlation
%         filename = sprintf('Slice%02d_Channel%02d_filecorrstruct.mat', sl, options.channel);
%         filecorrstruct = load(fullfile(options.metrics_folder, filename));

    %load trial-to-trial correlation
    filename = sprintf('Slice%02d_Channel%02d_trialcorrstruct.mat', sl, options.channel);
    trialcorrstruct = load(fullfile(options.metrics_folder, filename));

    %put into list
    pixel_corr_all = zeros(options.ntiffs,length(pixcorrstruct.file(1).mean_pixel_corr));
    for fidx=1:options.ntiffs
        pixel_corr_all(fidx,:)=pixcorrstruct.file(fidx).mean_pixel_corr;
    end

    %keep track of which slice and roi values came from 
    if sidx==1
        pixel_corr_roi = nanmean(pixel_corr_all,1);%average across file
        trial_corr_roi = max(trialcorrstruct.mean_trial_corr,[],2)';%take the max across stims
        %  file_corr_roi = filecorrstruct.mean_file_corr;
        slice_src = ones(1,size(pixel_corr_all,2))*double(sl);
        roi_src = 1:size(pixel_corr_all,2);
    else
        pixel_corr_roi = [pixel_corr_roi nanmean(pixel_corr_all,1)];%average across file
        trial_corr_roi = [trial_corr_roi max(trialcorrstruct.mean_trial_corr,[],2)'];%take the max across stims
       % file_corr_roi = [file_corr_roi filecorrstruct.mean_file_corr];
        slice_src =[slice_src ones(1,size(pixel_corr_all,2))*double(sl)];
        roi_src = [roi_src 1:size(pixel_corr_all,2)];
    end
end
trial_corr_roi2 = trial_corr_roi;
pixel_corr_roi2 = pixel_corr_roi;
slice_src2 = slice_src;
roi_src2 = roi_src;

%% load ROIs 3
roi_folder3 = 'manual2D_poly';%name of ROI subfolder
analysis_id = 'analysis03';

%load roiparams
roiparams_path = fullfile(A.roi_dir, roi_folder3, 'roiparams.mat');
roiparams = load(roiparams_path);

%create options structure
options = struct;
options.metrics_folder = fullfile(A.roi_dir, roi_folder3, 'metrics');
options.slices = [1];
options.ntiffs = A.ntiffs;
options.channel = A.signal_channel.(analysis_id);

%unpack ROI metrics-this could be turned into a fxn.....somehow
for sidx = 1:length(options.slices)
    sl = options.slices(sidx)

    %load pixel-to-pixel correlation
    filename = sprintf('Slice%02d_Channel%02d_pixcorrstruct.mat', sl, options.channel);
    pixcorrstruct = load(fullfile(options.metrics_folder, filename));

%         %load trial-to-trial correlation
%         filename = sprintf('Slice%02d_Channel%02d_filecorrstruct.mat', sl, options.channel);
%         filecorrstruct = load(fullfile(options.metrics_folder, filename));

    %load trial-to-trial correlation
    filename = sprintf('Slice%02d_Channel%02d_trialcorrstruct.mat', sl, options.channel);
    trialcorrstruct = load(fullfile(options.metrics_folder, filename));

    %put into list
    pixel_corr_all = zeros(options.ntiffs,length(pixcorrstruct.file(1).mean_pixel_corr));
    for fidx=1:options.ntiffs
        pixel_corr_all(fidx,:)=pixcorrstruct.file(fidx).mean_pixel_corr;
    end

    %keep track of which slice and roi values came from 
    if sidx==1
        pixel_corr_roi = nanmean(pixel_corr_all,1);%average across file
        trial_corr_roi = max(trialcorrstruct.mean_trial_corr,[],2)';%take the max across stims
        %  file_corr_roi = filecorrstruct.mean_file_corr;
        slice_src = ones(1,size(pixel_corr_all,2))*double(sl);
        roi_src = 1:size(pixel_corr_all,2);
    else
        pixel_corr_roi = [pixel_corr_roi nanmean(pixel_corr_all,1)];%average across file
        trial_corr_roi = [trial_corr_roi max(trialcorrstruct.mean_trial_corr,[],2)'];%take the max across stims
       % file_corr_roi = [file_corr_roi filecorrstruct.mean_file_corr];
        slice_src =[slice_src ones(1,size(pixel_corr_all,2))*double(sl)];
        roi_src = [roi_src 1:size(pixel_corr_all,2)];
    end
end
trial_corr_roi3 = trial_corr_roi;
pixel_corr_roi3 = pixel_corr_roi;
slice_src3 = slice_src;
roi_src3 = roi_src;


%% visualize distributions of individual variables
% [counts1,edges] = histcounts(pixel_corr_roi1);
% centers1 = (edges(1:end-1) + edges(2:end))/2;

[counts2, edges] = histcounts(pixel_corr_roi2);
%[counts2] = histcounts(pixel_corr_roi2,edges);
centers2 = (edges(1:end-1) + edges(2:end))/2;


[counts3] = histcounts(pixel_corr_roi3,edges);
centers3 = (edges(1:end-1) + edges(2:end))/2;


% %absolute counts
% figure;
% hold all
% bar(centers1,counts1,'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2,'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers3, counts3,'FaceColor',[1 0 0],'FaceAlpha',.5,'EdgeColor','none')
% xlabel('correlation')
% ylabel('count')
% legend('blobs','manual-circles','manual-polygon')
% title('pixel-to-pixel R')

%% distribution

% color1 = [1 0 0];
color2 = [0 1 0];
color3 = [0 0 1];
halfwidth = 0.015;
% halfheight = 0.005;
% roiname1 = strrep(roi_folder1, '_', '-');
roiname2 = strrep(roi_folder2, '_', '-');
roiname3 = strrep(roi_folder3, '_', '-');

figure;
hold all
% b = bar(centers1,counts1/size(pixel_corr_roi1,2), 'FaceColor','none','EdgeColor',color1)
% fill([b.XData-halfwidth; b.XData-halfwidth; b.XData+halfwidth; b.XData+halfwidth],...
%     [b.YData-halfheight; b.YData+halfheight; b.YData+halfheight; b.YData-halfheight], color1, 'FaceAlpha', 0.5)

b = bar(centers2,counts2/size(pixel_corr_roi2,2), 'FaceColor','none','EdgeColor','none')
fill([b.XData-halfwidth; b.XData-halfwidth; b.XData+halfwidth; b.XData+halfwidth],...
    [zeros(size(b.YData)); b.YData; b.YData; zeros(size(b.YData))], color2, 'FaceAlpha', 0.5)

b = bar(centers3,counts3/size(pixel_corr_roi3,2), 'FaceColor','none','EdgeColor', 'none')
fill([b.XData-halfwidth; b.XData-halfwidth; b.XData+halfwidth; b.XData+halfwidth],...
    [zeros(size(b.YData)); b.YData; b.YData; zeros(size(b.YData))], color3, 'FaceAlpha', 0.5)
% bar(centers1,counts1/size(pixel_corr_roi1,2),'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2/size(pixel_corr_roi2,2),'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers3, counts3/size(pixel_corr_roi3,2),'FaceColor',[1 0 0],'FaceAlpha',.5,'EdgeColor','none')

xlabel('correlation')
ylabel('p')
%legend('blobs1','blobs2','manual-polygon')
legend(roiname2, roiname3)
title('pixel-to-pixel R')

%visualize distributions of individual variables
% [counts1,edges] = histcounts(trial_corr_roi1);
% centers1 = (edges(1:end-1) + edges(2:end))/2;

[counts2] = histcounts(trial_corr_roi2,edges);
centers2 = (edges(1:end-1) + edges(2:end))/2;

[counts3] = histcounts(trial_corr_roi3,edges);
centers3 = (edges(1:end-1) + edges(2:end))/2;

% %absolute counts
% figure;
% hold all
% bar(centers1,counts1,'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2,'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers3, counts3,'FaceColor',[1 0 0],'FaceAlpha',.5,'EdgeColor','none')
% xlabel('correlation')
% ylabel('count')
% legend('blobs','manual-circles','manual-polygon')
% title('trial-to-trial R')

%distribution
figure;
hold all
% bar(centers1,counts1/size(trial_corr_roi1,2),'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2/size(trial_corr_roi2,2),'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers3, counts3/size(trial_corr_roi3,2),'FaceColor',[1 0 0],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers1,counts1/size(trial_corr_roi1,2),'FaceColor',[0 0 1],'EdgeColor','none')
bar(centers2, counts2/size(trial_corr_roi2,2),'FaceColor',[0 1 0],'EdgeColor','none')
bar(centers3, counts3/size(trial_corr_roi3,2),'FaceColor',[1 0 0],'EdgeColor','none')
xlabel('correlation')
ylabel('p')
% legend('blobs','manual-circles','manual-polygon')
legend(roi_folder2, roi_folder3)
title('trial-to-trial R')



%% plot both variables, for each roi
figure;
hold all
% plot(trial_corr_roi1,pixel_corr_roi1,'ob')%blob-detector
plot(trial_corr_roi2,pixel_corr_roi2,'og')%circles
plot(trial_corr_roi3,pixel_corr_roi3,'or')%polygon
hold off
legend(roiname2, roiname3)
% legend(roi_folder1, roi_folder2, roi_folder3)
xlabel('Trial-to-trial correlation')
ylabel('Pixel-to-pixel correlation')
axis('square')


%% project onto diagonal
% ref_vector = ones(size(pixel_corr_roi1,2),2);
% metric_combo_roi1 = dot([trial_corr_roi1' pixel_corr_roi1']', ref_vector') ./ sum(ref_vector .* ref_vector, 2)';%projection onto diagonal
ref_vector = ones(size(pixel_corr_roi2,2),2);
metric_combo_roi2 = dot([trial_corr_roi2' pixel_corr_roi2']', ref_vector') ./ sum(ref_vector .* ref_vector, 2)';%projection onto diagonal
ref_vector = ones(size(pixel_corr_roi3,2),2);
metric_combo_roi3 = dot([trial_corr_roi3' pixel_corr_roi3']', ref_vector') ./ sum(ref_vector .* ref_vector, 2)';%projection onto diagonal


% %visualize projections
% [counts1,edges] = histcounts(metric_combo_roi1);
% centers1 = (edges(1:end-1) + edges(2:end))/2;

[counts2,edges] = histcounts(metric_combo_roi2);
% [counts2] = histcounts(metric_combo_roi2,edges);
centers2 = (edges(1:end-1) + edges(2:end))/2;

[counts3] = histcounts(metric_combo_roi3,edges);
centers3 = (edges(1:end-1) + edges(2:end))/2;

%absolute counts
% figure;
% hold all
% bar(centers1,counts1,'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2,'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
%bar(centers3, counts3,'FaceColor',[1 0 0],'FaceAlpha',.5,'EdgeColor','none')
% xlabel('correlation')
% ylabel('count')
% legend('blobs','manual-circles')
% title('pixel-to-pixel R')

%distribution
figure;
hold all
% bar(centers1,counts1/size(metric_combo_roi1,2),'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2/size(metric_combo_roi2,2),'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers3, counts3/size(metric_combo_roi3,2),'FaceColor',[1 0 0],'FaceAlpha',.5,'EdgeColor','none')
%bar(centers1,counts1/size(metric_combo_roi1,2),'FaceColor',[0 0 1],'EdgeColor','none')
halfwidth2 = 0.015
b = bar(centers2, counts2/size(metric_combo_roi2,2),'FaceColor','none', 'EdgeColor', 'none');
fill([b.XData-halfwidth2; b.XData-halfwidth2; b.XData+halfwidth2; b.XData+halfwidth2],...
    [zeros(size(b.YData)); b.YData; b.YData; zeros(size(b.YData))], color2, 'FaceAlpha', 1)
halfwidth3 = 0.01
b = bar(centers3, counts3/size(metric_combo_roi3,2),'FaceColor', 'none', 'EdgeColor', 'none');
fill([b.XData-halfwidth3; b.XData-halfwidth3; b.XData+halfwidth3; b.XData+halfwidth3],...
    [zeros(size(b.YData)); b.YData; b.YData; zeros(size(b.YData))], color3, 'FaceAlpha', 1)

xlabel('correlation')
ylabel('p')
% legend('blobs','manual-circles','manual-polygon')
% legend(roi_folder1, roi_folder2, roi_folder3)
legend(roiname2, roiname3)
title('diagonal projection R')

%% Check threshold pass for combined metric:
combo_thresh = 0.14
thresh_pass_ind_combo2 = find(metric_combo_roi2>combo_thresh)
thresh_pass_ind_combo3 = find(metric_combo_roi3>combo_thresh)


%% set thresholds
pix_corr_thresh = 0.04;
trial_corr_thresh = 0.2;

%screen rois
thresh_pass_ind3  = find(((trial_corr_roi3>=trial_corr_thresh) | (pixel_corr_roi3 >= pix_corr_thresh))==1);
thresh_pass_ind2  = find(((trial_corr_roi2>=trial_corr_thresh) | (pixel_corr_roi2 >= pix_corr_thresh))==1);

propostion_passed = length(thresh_pass_ind)/size(pixel_corr_roi3,2)


%sort based on trial-to-trial correlation (manual2d-poly)

selected_roi_info = [slice_src3(thresh_pass_ind)' roi_src3(thresh_pass_ind)' pixel_corr_roi3(thresh_pass_ind)' trial_corr_roi3(thresh_pass_ind)' ...
    metric_combo_roi3(thresh_pass_ind)'];
[~, sorted_ind] = sort(selected_roi_info(:,4),1,'descend');
selected_roi_info_sorted = selected_roi_info(sorted_ind,:);
