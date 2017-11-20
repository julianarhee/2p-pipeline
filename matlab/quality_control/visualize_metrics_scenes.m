

clear all

% Set info manually:
source = '/nas/volume1/2photon/projects';
experiment = 'scenes';
session = '20171003_JW016';
acquisition = 'FOV1'; %'FOV1_zoom3x';
tiff_source = 'functional'; %'functional_subset';
acquisition_base_dir = fullfile(source, experiment, session, acquisition);


analysis_id = sprintf('analysis%02d',1);
% analysis_id = sprintf('analysis%02d',2);


%load reference file
path_to_reference = fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source));
A = load(path_to_reference);
if ~isfield(A,'roi_dir')
    A.roi_dir = fullfile(acquisition_base_dir,'ROIs');
end
roi_folder = 'blobs_DoG';%name of ROI subfolder
% roi_folder = 'blobs_DoG1';%name of ROI subfolder

%load roiparams
roiparams_path = fullfile(A.roi_dir, roi_folder, 'roiparams.mat');
roiparams = load(roiparams_path);

%create options structure
options = struct;
options.metrics_folder = fullfile(A.roi_dir, roi_folder, 'metrics');
options.slices = [5:5:41];
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
trial_corr_roi1 = trial_corr_roi;
pixel_corr_roi1 = pixel_corr_roi;
slice_src1 = slice_src;
roi_src1 = roi_src;


%% Load ROI2:
roi_folder = 'manual2D';%name of ROI subfolder
analysis_id1 = sprintf('analysis%02d',4);


% %load reference file
% path_to_reference = fullfile(acquisition_base_dir, sprintf('reference_%s.mat', tiff_source));
% A = load(path_to_reference);
% if ~isfield(A,'roi_dir')
%     A.roi_dir = fullfile(acquisition_base_dir,'ROIs');
% end

%load roiparams
roiparams_path = fullfile(A.roi_dir, roi_folder, 'roiparams.mat');
roiparams = load(roiparams_path);

%create options structure
options = struct;
options.metrics_folder = fullfile(A.roi_dir, roi_folder, 'metrics');
options.slices = [5:5:41];
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


%visualize distributions of individual variables
[counts1,edges] = histcounts(pixel_corr_roi1);
centers1 = (edges(1:end-1) + edges(2:end))/2;

[counts2] = histcounts(pixel_corr_roi2,edges);
centers2 = (edges(1:end-1) + edges(2:end))/2;

%absolute counts
figure;
hold all
% bar(centers1,counts1,'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2,'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
bar(centers1,counts1,'FaceColor',[0 0 1],'EdgeColor','none')
bar(centers2, counts2,'FaceColor',[0 1 0],'EdgeColor','none')
xlabel('correlation')
ylabel('count')
legend('blobs','manual-circles')
title('pixel-to-pixel R')

%distribution
figure;
hold all
% bar(centers1,counts1/size(pixel_corr_roi1,2),'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2/size(pixel_corr_roi2,2),'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
bar(centers1,counts1/size(pixel_corr_roi1,2),'FaceColor',[0 0 1],'EdgeColor','none')
bar(centers2, counts2/size(pixel_corr_roi2,2),'FaceColor',[0 1 0],'EdgeColor','none')
xlabel('correlation')
ylabel('p')
legend('blobs','manual-circles')
title('pixel-to-pixel R')

%visualize distributions of individual variables
[counts1,edges] = histcounts(trial_corr_roi1);
centers1 = (edges(1:end-1) + edges(2:end))/2;

[counts2] = histcounts(trial_corr_roi2,edges);
centers2 = (edges(1:end-1) + edges(2:end))/2;

%absolute counts
% figure;
% hold all
% bar(centers1,counts1,'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2,'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% xlabel('correlation')
% ylabel('count')
% legend('blobs','manual-circles')
% title('trial-to-trial R')

%distribution
figure;
hold all
% bar(centers1,counts1/size(trial_corr_roi1,2),'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2/size(trial_corr_roi2,2),'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
bar(centers1,counts1/size(trial_corr_roi1,2),'FaceColor',[0 0 1],'EdgeColor','none')
bar(centers2, counts2/size(trial_corr_roi2,2),'FaceColor',[0 1 0],'EdgeColor','none')
xlabel('correlation')
ylabel('p')
legend('blobs','manual-circles')
title('trial-to-trial R')



%% plot both variables, for each roi
figure;
hold all
plot(trial_corr_roi1,pixel_corr_roi1,'ok')%blob-detector
plot(trial_corr_roi2,pixel_corr_roi2,'or')%circles
hold off
legend('blobs','manual-circles')
xlabel('Trial-to-trial correlation')
ylabel('Pixel-to-pixel correlation')
axis('square')


%project onto diagonal
ref_vector = ones(size(pixel_corr_roi1,2),2);
metric_combo_roi1 = dot([trial_corr_roi1' pixel_corr_roi1']', ref_vector') ./ sum(ref_vector .* ref_vector, 2)';%projection onto diagonal
ref_vector = ones(size(pixel_corr_roi2,2),2);
metric_combo_roi2 = dot([trial_corr_roi2' pixel_corr_roi2']', ref_vector') ./ sum(ref_vector .* ref_vector, 2)';%projection onto diagonal
% for idx = 1:length(trial_corr_roi1,2)
% metric_combo_ro1(idx) = dot([trial_corr_roi1(idx) pixel_corr_roi1(idx)], ref_vector, 2) ./ sum(ref_vector .* ref_vector, 2);%projection onto diagonal
% metric_combo_ro2(idx) = dot([trial_corr_roi1(idx) pixel_corr_roi2(idx)], ref_vector, 2) ./ sum(ref_vector .* ref_vector, 2);%projection onto diagonal
% end


%visualize projections
[counts1,edges] = histcounts(metric_combo_roi1);
centers1 = (edges(1:end-1) + edges(2:end))/2;

[counts2] = histcounts(metric_combo_roi2,edges);
centers2 = (edges(1:end-1) + edges(2:end))/2;

%absolute counts
% figure;
% hold all
% bar(centers1,counts1,'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2,'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
% xlabel('correlation')
% ylabel('count')
% legend('blobs','manual-circles')
% title('diagonal projection')

%distribution
figure;
hold all
% bar(centers1,counts1/size(metric_combo_roi1,2),'FaceColor',[0 0 1],'FaceAlpha',.5,'EdgeColor','none')
% bar(centers2, counts2/size(metric_combo_roi2,2),'FaceColor',[0 1 0],'FaceAlpha',.5,'EdgeColor','none')
bar(centers1,counts1/size(metric_combo_roi1,2),'FaceColor',[0 0 1],'EdgeColor','none')
bar(centers2, counts2/size(metric_combo_roi2,2),'FaceColor',[0 1 0],'EdgeColor','none')
xlabel('correlation')
ylabel('p')
legend('blobs','manual-circles')
title('diagonal projection')


%set thresholds
pix_corr_thresh = 0.15;
trial_corr_thresh = 0.2;

%screen rois
thresh_pass_ind  = find(((trial_corr_roi2>=trial_corr_thresh) | (pixel_corr_roi2 >= pix_corr_thresh))==1);

%sort based on trial-to-trial correlation (manual2d)

selected_roi_info = [slice_src2(thresh_pass_ind)' roi_src2(thresh_pass_ind)' pixel_corr_roi2(thresh_pass_ind)' trial_corr_roi2(thresh_pass_ind)' ...
    metric_combo_roi2(thresh_pass_ind)'];

selected_roi_info = [slice_src1(thresh_pass_ind)' roi_src1(thresh_pass_ind)' pixel_corr_roi1(thresh_pass_ind)' trial_corr_roi1(thresh_pass_ind)' ...
    metric_combo_roi1(thresh_pass_ind)'];

% [~, sorted_ind] = sort(selected_roi_info(:,4),1,'descend');
[~, sorted_ind] = sort(selected_roi_info(:,5),1,'descend');
selected_roi_info_sorted = selected_roi_info(sorted_ind,:);

% save(fullfile(A.roi_dir,I.roi_id, 'metrics',filename),'-struct', 'selected_rois');

%%
combo_thresh_val = 0.14

combo_thresh1 = find(metric_combo_roi1>combo_thresh_val)
combo_thresh2 = find(metric_combo_roi2>combo_thresh_val)

roilist = zeros(length(combo_thresh2), 2);
for r=1:length(combo_thresh2)
    roilist(r, :) = [slice_src2(combo_thresh2(r)), roi_src2(combo_thresh2(r))];
end

roilist