
basepath = 'nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis';
datastruct = 'datastruct_002';

currPath = fullfile(basepath, datastruct, strcat(datastruct, '.mat'));

D = load(currPath);
meta = load(D.metaPath);

sidx = 8; % slice13
fidx = 4;
currsliceidx = 13;

filetiffs = dir(fullfile(meta.tiffPaths{fidx}, '*.tif'));
currtiffpath = fullfile(meta.tiffPaths{fidx}, filetiffs(currsliceidx).name);
Y = tiffRead(currtiffpath);
[d1, d2, d3] = size(Y);
avgY = mean(Y,3);
fov = repmat(mat2gray(avgY), [1, 1, 3]);

mapstruct = load(fullfile(D.outputDir, strcat(D.mapStructNames{sidx}, '.mat')));

phase = mapstruct.file(fidx).phase;
ratio = mapstruct.file(fidx).ratio;

thresh = 0.015;

currmap = threshold_map(phase, ratio, thresh);
figure(); 
axes();
imagesc(fov); 
hold on; 
imagesc2(currmap); 
colormap(gca, hsv);


% Try dilating:
SE = strel('square',2);
BW = im2bw(currmap);
currmapdilate = imdilate(BW, SE);
figure();
imshow(currmapdilate);
% -----


% Try smoothing pixel-map:
% currmapsmooth = imgaussfilt(currmap, 1);
% currmapsmooth = medfilt2(currmap, [2 2], 'symmetric');
currmapsmooth = medfilt2(currmapdilate, [2 2], 'symmetric');
figure(); subplot(1,2,1); imagesc2(currmap); colormap(hsv);
hold on; subplot(1,2,2); imagesc2(currmapsmooth); colormap(gray);

% diff smoothing options?
medfilt = ordfilt2(currmap,1,ones(2,2));
minfilt = ordfilt2(currmap,1,ones(2,2));
maxfilt = ordfilt2(currmap,1,ones(2,2));
cardfilt = ordfilt2(currmap,1,[0 1 0; 1 0 1; 0 1 0]);

figure();
subplot(2,2,1); imagesc2(medfilt); title('median');
subplot(2,2,2); imagesc2(minfilt); title('min');
subplot(2,2,3); imagesc2(maxfilt); title('max');
subplot(2,2,4); imagesc2(cardfilt); title('cardinal');

% -------


% Get entropy map (entropy val around 9x9 pixel neighborhood):
%E = entropyfilt(currmap);
E = entropyfilt(maxfilt);
Eim = mat2gray(E);
figure();
imshow(Eim);

% Threshold to segment threshold based on intensity value of pixels at
% boundaries:
% BW1 = imbinarize(Eim, 0.18);
olevel = graythresh(Eim);
BW1 = im2bw(Eim, olevel);
figure();
subplot(1,2,1)
ax=imagesc2(currmap); 
colormap(gca, hsv);
hold on;
subplot(1,2,2)
ax2=imshow(BW1);
colormap(gca, gray)
