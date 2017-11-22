function [tracesPath, nSlicesTrace] = get_traces_3D(D)


acquisitionName = D.acquisitionName;
nTiffs = D.nTiffs;
nchannels = D.channelIdx;
meta = load(D.metaPath);


% Create output dirs if no exist:
tracesPath = fullfile(D.datastructPath, 'traces');
if ~exist(tracesPath, 'dir')
    mkdir(tracesPath);
end

masksPath = fullfile(D.datastructPath, 'masks');
if ~exist(masksPath, 'dir')
    mkdir(masksPath)
end


% Load 3D mask mat and reshaped movie arrays to extract traces:

maskstruct = load(D.maskmatPath);

if D.average
    memfilepath = D.averagePath;
else
    memfilepath = D.mempath;
end

data_fns = dir(fullfile(memfilepath, '*.mat')); % Load memmaped movie files


for fidx=1:nTiffs

% Initialize output stucts:
maskStruct3D = struct();
T = struct();
    
data = matfile(fullfile(memfilepath, data_fns(fidx).name)); %This is equiv. to 'maskstruct' in 2d

sizY= size(data.Y);
d1 = sizY(1);
d2 = sizY(2);
d3 = sizY(3);


% Get raw traces using maskmay:
traces = maskstruct.maskmat' * double(data.Yr); % nRows = nROIS, nCols = tpoints


% Create "masks" from 3D spatial components:
roiMat = full(maskstruct.maskmat); 
nRois = size(roiMat,2);

% To view ROIs:
% if view_sample
%     maskarray = masks{1}; % roi=1
%     patch(isosurface(smooth3(maskarray)), 'FaceColor', 'blue', 'EdgeColor', 'none')
%     view(3); 
%     axis vis3d %tight
%     set(gca,'XLim',[1,d1],'YLim',[1,d2],'ZLim',[1,d3]);
%     camlight left; 
%     lighting gouraud
% end

maskcellTmp = arrayfun(@(roi) reshape(roiMat(:,roi), d1, d2, d3), 1:nRois, 'UniformOutput', false);
% if maskStruct.file(fidx).preprocessing.scaleFOV
%     maskcellTmp = arrayfun(@(roi) imresize(full(maskcellTmp{roi}), [size(maskcellTmp{roi},1)/2, size(maskcellTmp{roi},2)]), 1:nRois, 'UniformOutput', false);
% end
maskcell3D = cellfun(@logical, maskcellTmp, 'UniformOutput', false); % Does this work on sparse mats?
%maskcell = cellfun(@sparse, maskcell, 'UniformOutput', false);

nslices = d3;

roislices = cellfun(@(roimask) find(cell2mat(arrayfun(@(sl) any(find(roimask(:,:,sl))), 1:nslices, 'UniformOutput', 0))), maskcell3D, 'UniformOutput', 0);



% Get averages of each slice:
avgs = zeros([d1,d2,d3]);
for slice=1:d3
    avgs(:,:,slice) = mean(data.Y(:,:,slice,:), 4);
end

% for slice=1:d3
%     slicepath = '/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/memfiles/averaged/averagemem';
%     slicefn = sprintf('slice%02d.tif', slice);
%     tiffWrite(avgs(:,:,slice), slicefn, slicepath);
% end

% Get centers of each ROI:
coms = com(maskstruct.maskmat,d1,d2,d3);
if size(coms,2) == 2
    coms(:,3) = 1;
end
coms = round(coms);
if isfield(maskstruct, 'centroids')
    centers = maskstruct.centroids;
else
    centers = coms;
end


% Create "2D" masks for each slice, using centers:
currSliceRois = {};
nslices = d3;
for sl=1:nslices
    %currSliceRois{end+1} = find(centers(:,3)==sl);
    roi_ids_found = find(cell2mat(cellfun(@(rslices) any(find(rslices==sl)), roislices, 'UniformOutput', 0)));
    
    currSliceRois{end+1} = roi_ids_found; %maskstruct.roiIDs(roi_ids_found);
end

% TODO:  if go with 3D auto-ROIs, fix everything upstream so that we don't
% have to have silly re-organizing of all the ROIs and masks... Much faster
% and efficient to keep things in mat-form.

% This is stupid because it is just reshaping everything to follow
% structure of manually-selected ROIs to plot in roigui.m....
tic()
nrois = length(maskcell3D);
%maskcell = cell(1,nrois);
for sl=1:nslices

    mask_slicefn = sprintf('masks_Slice%02d.mat', D.slices(sl));
    maskPath = fullfile(D.datastructPath, 'masks', mask_slicefn);
    
    if exist(maskPath, 'file')
        maskStruct = load(maskPath);
    else
        maskStruct = struct();
    end
    
    ncurrrois = length(currSliceRois{sl});
    maskcell = cell(1,ncurrrois);
    for roi=1:length(currSliceRois{sl})%nrois
        roidx = currSliceRois{sl}(roi);
        %maskcell{roi} = maskcell3D{roidx}(:,:,centers(roidx, 3));
        maskcell{roi} = maskcell3D{roidx}(:,:,sl);
    end
    
    maskStruct.file(fidx).maskcell = maskcell;
    maskStruct.file(fidx).centers = centers(currSliceRois{sl},:); %currSliceRois{sl};
    maskStruct.file(fidx).roi3Didxs = maskstruct.roiIDs(currSliceRois{sl}); %maskstruct.roiIDs(roi_ids_found);
    
    %mask_slicefn = sprintf('masks_Slice%02d_File%03d.mat', sl, fidx);
    %mask_slicefn = sprintf('masks_Slice%02d.mat');
    %maskPath = fullfile(D.datastructPath, 'masks', mask_slicefn);
    [fp,fn,fe] = fileparts(maskPath);
    
    if exist(maskPath, 'file')
        save_struct(fp, strcat(fn,fe), maskStruct, '-append')
    else
        save_struct(fp, strcat(fn,fe), maskStruct)
    end
    
    % Also extract traces for curr-slice masks:
    tracesName = sprintf('traces_Slice%02d_Channel%02d.mat', D.slices(sl), 1);
    if exist(fullfile(tracesPath, tracesName), 'file')
        tracestruct = load(fullfile(tracesPath, tracesName));
    else
        tracestruct = struct();
    end
    tracestruct.file(fidx).rawTraces = traces(currSliceRois{sl},:)'; %rawTraces;
    

    %T.masks.file(fidx) = {masks};
    tracestruct.file(fidx).avgImage = avgs(:,:,sl);
    %T.file(fidx).slicePath = fullfile(slicePath, currSliceName);
    %T.file(fidx).badFrames = badframes;
    %T.file(fidx).corrcoeffs = corrs;
    %T.file(fidx).refframe = refframe;
    tracestruct.file(fidx).info.szFrame = sizY(1:2); %size(avgY);
    tracestruct.file(fidx).info.nFrames = sizY(:,end);
    tracestruct.file(fidx).info.nRois = length(currSliceRois{sl}); %size(tr,1); %length(maskcell);

    if exist(fullfile(tracesPath, tracesName), 'file')
        save_struct(tracesPath, tracesName, tracestruct, '-append');
    else
        fprintf('Saving struct: %s.\n', tracesName);
        save_struct(tracesPath, tracesName, tracestruct);
    end

end

    
% Save 3D mask info for current movie:
% ------------------------------------
maskStruct3D.maskcell3D = maskcell3D;
maskStruct3D.centers = centers;
maskStruct3D.coms = coms;
maskStruct3D.roiMat = roiMat;
maskStruct3D.roi3Didxs = maskstruct.roiIDs;

% This maskPath points to the 3D mask cell array, i.e., each cell contains
% 3D mask, and length of cell array is nRois:
maskpath3D = fullfile(D.datastructPath, 'masks', sprintf('manual3D_masks_File%03d.mat', fidx));

[fp,fn,fe] = fileparts(maskpath3D);
%save_struct(fp, strcat(fn,fe), maskStruct3D)
save(maskpath3D, '-struct', 'maskStruct3D','-v7.3');


fprintf('Saving struct: %s.\n', maskpath3D);

%else

% Save 3D traces info for current movie:
% ------------------------------------
T.rawTraces = traces'; %rawTraces;

% Include nmf outptu infered traces:
% inferredTraces = arrayfun(@(i) tmpC(i,:)/tmpDf(i), 1:nrois, 'UniformOutput', 0);
% inferredTraces = cat(1, inferredTraces{1:end});
% T.inferredTraces = inferredTraces';

%T.masks.file(fidx) = {masks};
T.avgImage = avgs;
T.tiffPath = memfilepath;
%T.file(fidx).badFrames = badframes;
%T.file(fidx).corrcoeffs = corrs;
%T.file(fidx).refframe = refframe;
T.info.szFrame = sizY(1:2); %size(avgY);
T.info.nFrames = sizY(end);
T.info.nRois = length(maskStruct3D); %length(maskcell);


fprintf('Extracted traces for %i of %i FILES.\n', fidx, nTiffs);

tracesName = sprintf('traces3D_Channel%02d_File%03d', 1, fidx);
fprintf('Saving struct: %s.\n', tracesName);

save_struct(tracesPath, tracesName, T);

% 
tracesSaved = dir(fullfile(tracesPath, 'traces_Slice*'));
tracesSaved = tracesSaved(arrayfun(@(x) ~strcmp(x.name(1),'.'),tracesSaved));
nSlicesTrace = length(tracesSaved);


%   
end
