% RETINOTOPY.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

% acquisition_info;
%session = '20161219_JR030W';
%session = '20161221_JR030W';
%session = '20161218_CE024';
session = '20161222_JR030W'

%experiment = 'retinotopy2';
%experiment = 'test_crossref';
%experiment = 'retinotopyFinalMask';
%experiment = 'retinotopyFinal';
%experiment = 'retinotopyControl';
experiment = 'retinotopy1'
%experiment = 'test_crossref/nmf';

%analysis_no = 17 %16 %15 %13 %13 %9 %7;
analysis_no = 1 %3 %4
tefo = false; %true;

D = loadAnalysisInfo(session, experiment, analysis_no, tefo);


%trimEnd = true;
slicesToUse = D.slices;

meta = load(D.metaPath)
nTiffs = meta.nTiffs;

% -------------------------------------------------------------------------
% Process traces for FFT analysis:
% -------------------------------------------------------------------------
% for sidx = 1:length(slicesToUse)
% 
%     currSlice = slicesToUse(sidx);
%     fprintf('Processing traces for Slice %02d...\n', currSlice);
%     
% Assumes all TIFFs are reps of e/o, so just use file1:
targetFreq = meta.file(1).mw.targetFreq;
winUnit = (1/targetFreq);
crop = meta.file(1).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
nWinUnits = 3;
switch D.roiType
    case '3Dcnmf'
        processTraces3Dnmf(D, winUnit, nWinUnits, crop)
    case 'manual3Drois'
        processTraces3Dnmf(D, winUnit, nWinUnits, crop)
    case 'cnmf'
        % do other stuff
    otherwise
        processTraces(D, winUnit, nWinUnits, crop)
end

% end

% -------------------------------------------------------------------------
% Get DF/F for whole movie:
% -------------------------------------------------------------------------
dfMin = 5; %0;
switch D.roiType
    case '3Dcnmf'
        dfstruct = getDfMovie3Dnmf(D, dfMin);
    case 'manual3Drois'
        dfstruct = getDfMovie3Dnmf(D, dfMin);
    case 'cnmf'
        % do other stuff
    otherwise
        dfstruct = getDfMovie(D, dfMin);
end
D.dfStructName = dfstruct.name;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');


% Get legends:
if ~isfield(meta, 'legends')
    fprintf('Creating legends.\n');
    legends = makeLegends(D.outputDir);
    meta.legends = legends;
    save(D.metaPath, '-append', '-struct', 'meta');
end

if ~isfield(meta, 'cmaps')
    condtypes = fieldnames(meta.legends);
    figure();
    for legend = 1:length(condtypes)
        imagesc(meta.legends.(condtypes{legend})); caxis([-pi, pi]); cmap = colormap(hsv);
        meta.cmaps.(condtypes{legend}) = cmap;
    end
    save(D.metaPath, '-append', '-struct', 'meta');
end
%% 3D MAPS!
% =========================================================================
% With 3D ROIs, assign a metric to 3D cell (instead of by slice):
% =========================================================================

% IF 3D rois, use 3D traces, before splitting into slices:
fstart = tic();

% -------------------------------------------------------------------------
% FFT analysis:
% -------------------------------------------------------------------------
% For each analysed slice, do FFT analysis on each file (i.e., run, or
% condition rep).
% For a given set of traces (on a given run) extracted from a slice, use 
% corresponding metaInfo for maps.

mapStructNames3D = cell(1,nTiffs);
fftStructNames3D = cell(1,nTiffs);

maskPaths3D = D.maskInfo.maskPaths;

% Load tracestruct:
for fidx=1:nTiffs
    tracestruct = load(fullfile(D.tracesPath, D.traceNames3D{fidx}));
    
    fftStruct = struct();

    fprintf('Processing TIFF #%i...\n', fidx);

    sliceIdxs = slicesToUse(1):meta.file(fidx).si.nFramesPerVolume:meta.file(fidx).si.nTotalFrames;

    %expectedTimes = linspace(0, meta.file(fidx).mw.mwDur, meta.file(fidx).mw.nTrueFrames);
    %expectedTImes = expectedTimes + meta.file(fidx).mw.mwSec(1); % add offset between trigger and stim-display

%     if isfield(tracestruct.file(fidx), 'inferredTraces')
%         inferredTraces = tracestruct.file(fidx).inferredTraces;
%         inferred = true;
%     else
%       inferred = false;
%     end
    if isfield(tracestruct, 'dfTracesNMF')
        %dfTraceMatNMF = tracestruct.dfTraceMatNMF;
        %detrendedTracesNMF = tracestruct.detrendedTraceMatNMF;
        inferred = true;
    else
        inferred = false;
    end
    
    %traceMatDC = tracestruct.traceMatDC;
    %DCs = tracestruct.DCs;
    %tmptraces = bsxfun(@minus, traceMatDC, DCs);
    traces = tracestruct.traceMat;
        
    targetFreq = meta.file(fidx).mw.targetFreq;
    nCycles = meta.file(fidx).mw.nCycles;
    nTotalSlices = meta.file(fidx).si.nFramesPerVolume;

    %crop = meta.file(fidx).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
    volsize = meta.volumeSizePixels;
    [nframes,nrois] = size(traces);
            

        
    % Do FFT on each ROI's trace:
    Fs = meta.file(fidx).si.siVolumeRate;
    N = size(traces,1);
    dt = 1/Fs;
    t = dt*(0:N-1)';
    dF = Fs/N;
    freqs = dF*(0:N/2-1)';
    freqIdx = find(abs((freqs-targetFreq))==min(abs(freqs-targetFreq)));

    fftfun = @(x) fft(x)/N;
    fftMat = arrayfun(@(i) fftfun(traces(:,i)), 1:size(traces,2), 'UniformOutput', false);
    fftMat = cat(2, fftMat{1:end});
  
    fftMat = fftMat(1:N/2, :);
    fftMat(2:end,:) = fftMat(2:end,:).*2;

    magMat = abs(fftMat);
    ratioMat = magMat(freqIdx,:) ./ (sum(magMat, 1) - magMat(freqIdx,:));
    phaseMat = angle(fftMat);

    [maxmag,maxidx] = max(magMat);
    maxFreqs = freqs(maxidx);
    

    % Get FFT-MAT for inferred: ---------------------------------
    if inferred
        % Use raw fluor matrix from NMF:
        tracesNMF = tracestruct.rawTraceMatNMF;
        fftMatNMF = arrayfun(@(i) fftfun(tracesNMF(:,i)), 1:size(tracesNMF,2), 'UniformOutput', false);
        fftMatNMF = cat(2, fftMatNMF{1:end});
        fftMatNMF = fftMatNMF(1:N/2, :);
        fftMatNMF(2:end,:) = fftMatNMF(2:end,:).*2;
        magMatNMF = abs(fftMatNMF);
        ratioMatNMF = magMatNMF(freqIdx,:) ./ (sum(magMatNMF, 1) - magMatNMF(freqIdx,:));
        phaseMatNMF = angle(fftMatNMF);
        [maxmag2,maxidx2] = max(magMatNMF);
        maxFreqsNMF = freqs(maxidx2);

        % Use output of NMF:
        tracesNMFoutput = tracestruct.detrendedTraceMatNMF;
        fftMatNMFoutput = arrayfun(@(i) fftfun(tracesNMFoutput(:,i)), 1:size(tracesNMFoutput,2), 'UniformOutput', false);
        fftMatNMFoutput = cat(2, fftMatNMFoutput{1:end});
        fftMatNMFoutput = fftMatNMFoutput(1:N/2, :);
        fftMatNMFoutput(2:end,:) = fftMatNMFoutput(2:end,:).*2;
        magMatNMFoutput = abs(fftMatNMFoutput);
        ratioMatNMFoutput = magMatNMFoutput(freqIdx,:) ./ (sum(magMatNMFoutput, 1) - magMatNMFoutput(freqIdx,:));
        phaseMatNMFoutput = angle(fftMatNMFoutput);
        [maxmag3,maxidx3] = max(magMatNMFoutput);
        maxFreqsNMFoutput = freqs(maxidx2);

    end
    % ------------------------------------------------------------

        
    fftStruct.targetPhase = phaseMat(freqIdx,:);
    fftStruct.targetMag = magMat(freqIdx,:);
    fftStruct.freqsAtMaxMag = maxFreqs;
    fftStruct.fftMat = fftMat;
    fftStruct.traces = traces;
    fftStruct.sliceIdxs = sliceIdxs;
    fftStruct.targetFreq = targetFreq;
    fftStruct.freqs = freqs;
    fftStruct.targetFreqIdx = freqIdx;
    fftStruct.magMat = magMat;
    fftStruct.ratioMat = ratioMat;
    fftStruct.phaseMat = phaseMat;

    if inferred
        fftStruct.targetPhaseNMF = phaseMatNMF(freqIdx,:);
        fftStruct.targetMagNMF = magMatNMF(freqIdx,:);
        fftStruct.freqsAtMaxMagNMF = maxFreqsNMF;
        fftStruct.fftMatNMF = fftMatNMF;
        fftStruct.tracesNMF = tracesNMF; %detrendedTraceMatNMF;
        fftStruct.magMatNMF = magMatNMF;
        fftStruct.ratioMatNMF = ratioMatNMF;
        fftStruct.phaseMatNMF = phaseMatNMF;


        fftStruct.targetPhaseNMFoutput = phaseMatNMFoutput(freqIdx,:);
        fftStruct.targetMagNMFoutput = magMatNMFoutput(freqIdx,:);
        fftStruct.freqsAtMaxMagNMFoutput = maxFreqsNMFoutput;
        fftStruct.fftMatNMFoutput = fftMatNMFoutput;
        fftStruct.tracesNMFoutput = tracesNMFoutput; %detrendedTraceMatNMF;
        fftStruct.magMatNMFoutput = magMatNMFoutput;
        fftStruct.ratioMatNMFoutput = ratioMatNMFoutput;
        fftStruct.phaseMatNMFoutput = phaseMatNMFoutput;
    end



        
        % Get MAPS for current slice: ---------------------------------
        tic();
        
%         % TOD:  Make 3D maps??
% 
%         phaseMap = assignRoiMap(maskcell, phaseMap, phaseMat, freqIdx);
%         magMap = assignRoiMap(maskcell, magMap, magMat, freqIdx);
%         ratioMap = assignRoiMap(maskcell, ratioMap, ratioMat);
%         phaseMaxMag = assignRoiMap(maskcell, phaseMat, magMat, [], freqs);
% 
% 
%         % --
%         if inferred
%             phaseMapInferred = assignRoiMap(maskcell, phaseMapInferred, phaseMatInferred, freqIdx);
%             magMapInferred = assignRoiMap(maskcell, magMapInferred, magMatInferred, freqIdx);
%             ratioMapInferred = assignRoiMap(maskcell, ratioMapInferred, ratioMatInferred);
%             phaseMaxMagInferred = assignRoiMap(maskcell, phaseMatInferred, magMatInferred, [], freqs);
%         end
%         % ---
%         
%         toc();
%         
%         maps.file(fidx).magnitude = magMap;
%         maps.file(fidx).phase = phaseMap;
%         maps.file(fidx).phasemax = phaseMaxMag;
%         maps.file(fidx).ratio = ratioMap;
%         maps.file(fidx).avgY = avgY;
%         
%         % ---
%         if inferred
%             maps.file(fidx).magnitudeInferred = magMapInferred;
%             maps.file(fidx).phaseInferred = phaseMapInferred;
%             maps.file(fidx).phasemaxInferred = phaseMaxMagInferred;
%             maps.file(fidx).ratioInferred = ratioMapInferred;
%         end
%         % ---

        % --- Need to reshape into 2d image if using pixels:
%         if strcmp(D.roiType, 'pixels')
%             mapTypes = fieldnames(maps);
%             for map=1:length(mapTypes)
%                 currMap = maps.(mapTypes{map});
%                 currMap = reshape(currMap, [d1, d2, size(currMap,3)]);
%                 maps.(mapTypes{map}) = currMap;
%             end
%         end
        % --------------------------------------------------------------
        
    
%     % Save maps for current slice:
%     mapStructName = sprintf('maps3D_File%03d', fidx);
%     save_struct(D.outputDir, mapStructName, maps);

    fftStructName = sprintf('fft3D_File%03d', fidx);
    save_struct(D.outputDir, fftStructName, fftStruct);

    %M.file(fidx) = maps;
    
    fprintf('Finished FFT analysis for File %03d.\n', fidx);
    
%     mapStructNames3D{fidx} = mapStructName;
    fftStructNames3D{fidx} = fftStructName;
end

clear fftStruct maps T

fprintf('TOTAL TIME ELAPSED:\n');
toc(fstart);

%D.mapStructNames3D = mapStructNames3D;
D.fftStructNames3D = fftStructNames3D;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');


%% 3D VISUALIZE:
% =========================================================================
% VISUALIZE a given metric across (some of) the volume:
% =========================================================================

if strfind(D.roiType, '3D')
for fidx=1:length(meta.file)
%fidx = 4;
roi = 1;

volsize = meta.volumeSizePixels;
    
fftstruct = load(fullfile(D.outputDir, D.fftStructNames3D{fidx}));
%fftstruct = fftmat.file(fidx);

condtypes = fieldnames(meta.cmaps);
metric = 'ratio';
condmatch = find(cellfun(@(i) ~isempty(strfind(meta.file(fidx).mw.runName, i)), condtypes))
if numel(condmatch)>1
    condmatch = condmatch(1)
end
currcond = condtypes{condmatch};

if strcmp(D.roiType, '3Dcnmf')
inputSource = 'NMF' %output'
else
inputSource = ''
end

switch inputSource
    case 'NMF'
        targetphase = fftstruct.targetPhaseNMF;
        ratiomat = fftstruct.ratioMatNMF;
    case 'NMFoutput'
        targetphase = fftstruct.targetPhaseNMFoutput;
        ratiomat = fftstruct.ratioMatNMFoutput;
    otherwise
        targetphase = fftstruct.targetPhase;
        ratiomat = fftstruct.ratioMat;
end


switch metric
    case 'phase'
        currcmap = meta.cmaps.(currcond);
        cmapspace = linspace(-pi, pi, size(currcmap,1));
    case 'ratio'
        currcmap = colormap(hot);
        cmapspace = linspace(min(ratioMat), max(ratiomat), size(currcmap, 1));
end


% Test viewing:
maskPaths3D = D.maskInfo.maskPaths;
maskmat = load(maskPaths3D{fidx});
if strcmp(D.roiType, '3Dcnmf') || strcmp(D.roiType, '3Dnmf')
    masks = maskmat.spatialcomponents;
elseif strcmp(D.roiType, 'manual3Drois')
    masks = maskmat.roiMat;
end
nrois = size(masks,2);
clear maskmat

step=round(nrois/100)

for roi=1:step:nrois
    %rmapidx = find(abs(ratiospace-ratioMat(roi))==min(abs(ratiospace-ratioMat(roi))));
    switch metric
        case 'phase'
            if isnan(fftstruct.targetPhase(roi))
                continue;
            end
            cmapidx = find(abs(cmapspace-targetphase(roi))==min(abs(cmapspace-targetphase(roi))));

        case 'ratio'
            if isnan(fftstruct.ratioMat(roi))
                continue;
            end
            cmapidx = find(abs(cmapspace-ratiomat(roi))==min(abs(cmapspace-ratiomat(roi))));

    end
    if length(cmapidx)>1
        cmapidx = cmapidx(1);
    end
    mask = reshape(full(masks(:,roi)), volsize); %.*(fft.ratioMat(roi)/max(fft.ratioMat));
    data = smooth3(mask);
    %patch(isocaps(data,0.25*max(data(:))),'FaceColor',hotmap(rmapidx,:),'EdgeColor','none');
    [f,v] = isosurface(data, 0.25*max(data(:)));
    p1 = patch('Vertices', v, 'Faces', f, 'FaceColor', currcmap(cmapidx,:), 'EdgeColor', 'none'); %, 'FaceAlpha', ratioMat(roi)/max(ratioMat));
    isonormals(data,p1)
    hold all;
    %view(3);
    view(-15, 30);
    axis vis3d tight
    
    switch metric
        case 'phase'
            colormap(hsv)
            caxis([-pi pi])
        otherwise
            colormap(hot)
            caxis([min(cmapspace), max(cmapspace)])
    end

    drawnow;
end
%colorbar()
camlight headlight; 
lighting gouraud;
camlight left; 
lighting flat;
if strcmp(metric, 'ratio')
    colorbar();
end
set(gca,'zdir','reverse')
set(gca,'ydir','reverse')
set(gca,'xdir','normal')
set(gca, 'color', 'none')
set(gca, 'zlim', [0 22])

set(gca, 'ylim', [0 120])
set(gca, 'xlim', [0 120])

camlight headlight; 
lighting gouraud;

savefig(fullfile(D.figDir, sprintf('ratio3Dmap_File004_%s.fig', inputSource)))
saveas(p1, fullfile(D.figDir, sprintf('ratio3Dmap_File004_%s.png', inputSource)), 'png')

clf;

end

end

%% 2D maps (per slice):
% =========================================================================
% For roigui, look by SLICE, not be ROI, so parse each "slice" based on COM
% or centroid, and color that spot with metric.  All traces are taken as
% '3D' traces (not by slice) for 3D rois.
% =========================================================================

% TODO; instead of using COM or center for 2D, just assign any mask on a
% slice with a given cell's metric if that cell happens to span multiple
% slices.


fstart = tic();
% -------------------------------------------------------------------------
% FFT analysis:
% -------------------------------------------------------------------------
% For each analysed slice, do FFT analysis on each file (i.e., run, or
% condition rep).
% For a given set of traces (on a given run) extracted from a slice, use 
% corresponding metaInfo for maps.

mapStructNames = cell(1,length(slicesToUse));
fftStructNames = cell(1,length(slicesToUse));
for sidx = 1:length(slicesToUse)

    currSlice = slicesToUse(sidx); % - slicesToUse(1) + 1;
    fprintf('Processing Slice %02d of %i slices...\n', currSlice, length(slicesToUse));
    
    % Load masks:
    if ~strcmp(D.roiType, 'pixels')
        maskstruct = load(D.maskPaths{sidx});
    end
    
    % Load tracestruct:
    tracestruct = load(fullfile(D.tracesPath, D.traceNames{sidx}));
    
    fftStruct = struct();
    for fidx=1:nTiffs
        %fprintf('Processing TIFF #%i...\n', fidx);
        
        if ~strcmp(D.roiType, 'pixels')
            if isfield(maskstruct, 'file')
                maskcell = maskstruct.file(fidx).maskcell;
            else
                maskcell = maskstruct.maskcell;
            end
        end
        if isempty(maskcell)
            fprintf('Empty mask!\n')
            fftStruct.file(fidx).targetPhase = zeros(1,100);
            fftStruct.file(fidx).targetMag = zeros(1,100);
            fftStruct.file(fidx).freqsAtMaxMag = zeros(1,100);
            fftStruct.file(fidx).fftMat = zeros(1,100);
            fftStruct.file(fidx).traces = zeros(1,100);
            fftStruct.file(fidx).sliceIdxs = zeros(1,100);
            fftStruct.file(fidx).targetFreq = 1;
            fftStruct.file(fidx).freqs = zeros(1,100);
            fftStruct.file(fidx).targetFreqIdx = 1;
            fftStruct.file(fidx).magMat = zeros(1,100);
            fftStruct.file(fidx).ratioMat = zeros(1,100);
            fftStruct.file(fidx).phaseMat = zeros(1,100);
            
            maps.file(fidx).magnitude = zeros(1,100);
            maps.file(fidx).phase = zeros(1,100);
            maps.file(fidx).phasemax = zeros(1,100);
            maps.file(fidx).ratio = zeros(1,100);
            maps.file(fidx).avgY = zeros(1,100);

            if inferred
                fftStruct.file(fidx).targetPhaseNMF = zeros(1,100); 
                fftStruct.file(fidx).targetMagNMF = zeros(1,100);
                fftStruct.file(fidx).freqsAtMaxMagNMF = zeros(1,100);
                fftStruct.file(fidx).fftMatNMF = zeros(1,100); 
                fftStruct.file(fidx).tracesNMF = zeros(1,100); %detrendedTraceMatNMF;
                fftStruct.file(fidx).magMatNMF = zeros(1,100);
                fftStruct.file(fidx).ratioMatNMF = zeros(1,100);
                fftStruct.file(fidx).phaseMatNMF = zeros(1,100);

                fftStruct.file(fidx).targetPhaseNMFoutput = zeros(1,100);
                fftStruct.file(fidx).targetMagNMFoutput = zeros(1,100);
                fftStruct.file(fidx).freqsAtMaxMagNMFoutput = zeros(1,100) 
                fftStruct.file(fidx).fftMatNMFoutput = zeros(1,100); 
                fftStruct.file(fidx).tracesNMFoutput = zeros(1,100); %detrendedTraceMatNMF;
                fftStruct.file(fidx).magMatNMFoutput = zeros(1,100); 
                fftStruct.file(fidx).ratioMatNMFoutput = zeros(1,100); 
                fftStruct.file(fidx).phaseMatNMFoutput = zeros(1,100); 
 
                maps.file(fidx).magnitudeNMF = zeros(1,100); 
                maps.file(fidx).phaseNMF = zeros(1,100);
                maps.file(fidx).phasemaxNMF = zeros(1,100);
                maps.file(fidx).ratioNMF = zeros(1,100);

                maps.file(fidx).magnitudeNMFoutput = zeros(1,100); 
                maps.file(fidx).phaseNMFoutput = zeros(1,100);
                maps.file(fidx).phasemaxNMFoutput = zeros(1,100);
                maps.file(fidx).ratioNMFoutput = zeros(1,100);

            end




        else
                
            %sliceIdxs = currSlice:meta.file(fidx).si.nFramesPerVolume:meta.file(fidx).si.nTotalFrames;
            sliceIdxs = slicesToUse(sidx):meta.file(fidx).si.nFramesPerVolume:meta.file(fidx).si.nTotalFrames;

            expectedTimes = linspace(0, meta.file(fidx).mw.mwDur, meta.file(fidx).mw.nTrueFrames);
            expectedTImes = expectedTimes + meta.file(fidx).mw.mwSec(1); % add offset between trigger and stim-display

            %traces = traceStruct.traces.file{fidx};
%             if isfield(tracestruct.file(fidx), 'inferredTraces')
%                 inferredTraces = tracestruct.file(fidx).inferredTraces;
%                 inferred = true;
%             else
%                 inferred = false;
%             end
            if isfield(tracestruct.file(fidx), 'dfTracesNMF')
                %dfTraceMatNMF = tracestruct.file(fidx).dfTraceMatNMF;
                %detrendedTraceMatNMF = tracestruct.file(fidx).detrendedTraceMatNMF;
                inferred = true;
            else
                inferred = false;
            end

            %traceMatDC = tracestruct.file(fidx).traceMatDC;
            %DCs = tracestruct.file(fidx).DCs;
            %tmptraces = bsxfun(@minus, traceMatDC, DCs);
            traces = tracestruct.file(fidx).traceMat;

            if iscell(tracestruct.file(fidx).avgImage)
                avgY = tracestruct.file(fidx).avgImage{1};
            else
                avgY = tracestruct.file(fidx).avgImage;
            end
        
            targetFreq = meta.file(fidx).mw.targetFreq;
            nCycles = meta.file(fidx).mw.nCycles;
            nTotalSlices = meta.file(fidx).si.nFramesPerVolume;

            %crop = meta.file(fidx).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);

            switch D.roiType
                case 'create_rois'
                    [d1,d2] = size(avgY);
                    [nframes,nrois] = size(traces);
                case 'condition'
                    [d1,d2] = size(avgY);
                    [nframes,nrois] = size(traces);
                case 'pixels'
                    %[d1,d2,tpoints] = size(T.traces.file{fidx});
                    [d1, d2] = size(avgY);
                    nframes = size(traces,1);
                    nrois = d1*d2;
                case 'cnmf'
                    [d1,d2] = size(avgY);
                    [nframes,nrois] = size(traces);
                case '3Dcnmf'
                    [d1,d2,d3] = size(avgY);
                    [nframes,nrois] = size(traces);
                case 'manual3Drois'
                    [d1,d2,d3] = size(avgY);
                    [nframes,nrois] = size(traces);
            end

            % Get phase and magnitude maps:
            phaseMap = zeros([d1, d2]);
            magMap = zeros([d1, d2]);
            ratioMap = zeros([d1, d2]);
            phaseMaxMag = zeros([d1, d2]);

            % ---
            if inferred
                phaseMapNMF = zeros([d1, d2]);
                magMapNMF = zeros([d1, d2]);
                ratioMapNMF = zeros([d1, d2]);
                phaseMaxMagNMF = zeros([d1, d2]);

                phaseMapNMFoutput = zeros([d1, d2]);
                magMapNMFoutput = zeros([d1, d2]);
                ratioMapNMFoutput = zeros([d1, d2]);
                phaseMaxMagNMFoutput = zeros([d1, d2]);


            end
            % ---
            % Do FFT on each row:
            Fs = meta.file(fidx).si.siVolumeRate;
            N = size(traces,1);
            dt = 1/Fs;
            t = dt*(0:N-1)';
            dF = Fs/N;
            freqs = dF*(0:N/2-1)';
            freqIdx = find(abs((freqs-targetFreq))==min(abs(freqs-targetFreq)));

            fftfun = @(x) fft(x)/N;
            fftMat = arrayfun(@(i) fftfun(traces(:,i)), 1:size(traces,2), 'UniformOutput', false);
            fftMat = cat(2, fftMat{1:end});
      
            fftMat = fftMat(1:N/2, :);
            fftMat(2:end,:) = fftMat(2:end,:).*2;

            magMat = abs(fftMat);
            ratioMat = magMat(freqIdx,:) ./ (sum(magMat, 1) - magMat(freqIdx,:));
            phaseMat = angle(fftMat);

            [maxmag,maxidx] = max(magMat);
            maxFreqs = freqs(maxidx);


            % Get FFT-MAT for inferred: ---------------------------------
            if inferred
                % Do per-slice on raw fluorescent mat from NMF:
                tracesNMF = tracestruct.file(fidx).rawTraceMatNMF;
                fftMatNMF = arrayfun(@(i) fftfun(tracesNMF(:,i)), 1:size(tracesNMF,2), 'UniformOutput', false);
                fftMatNMF = cat(2, fftMatNMF{1:end});

                fftMatNMF = fftMatNMF(1:N/2, :);
                fftMatNMF(2:end,:) = fftMatNMF(2:end,:).*2;

                magMatNMF = abs(fftMatNMF);
                ratioMatNMF = magMatNMF(freqIdx,:) ./ (sum(magMatNMF, 1) - magMatNMF(freqIdx,:));
                phaseMatNMF = angle(fftMatNMF);

                [maxmag2,maxidx2] = max(magMatNMF);
                maxFreqsNMF = freqs(maxidx2);


                % Do per-slice on NMF output:
                tracesNMFoutput = tracestruct.file(fidx).detrendedTraceMatNMF;
                fftMatNMFoutput = arrayfun(@(i) fftfun(tracesNMFoutput(:,i)), 1:size(tracesNMFoutput,2), 'UniformOutput', false);
                fftMatNMFoutput = cat(2, fftMatNMFoutput{1:end});

                fftMatNMFoutput = fftMatNMFoutput(1:N/2, :);
                fftMatNMFoutput(2:end,:) = fftMatNMFoutput(2:end,:).*2;

                magMatNMFoutput = abs(fftMatNMFoutput);
                ratioMatNMFoutput = magMatNMFoutput(freqIdx,:) ./ (sum(magMatNMFoutput, 1) - magMatNMFoutput(freqIdx,:));
                phaseMatNMFoutput = angle(fftMatNMFoutput);

                [maxmag3,maxidx3] = max(magMatNMFoutput);
                maxFreqsNMFoutput = freqs(maxidx3);



            end
            % ------------------------------------------------------------


            fftStruct.file(fidx).targetPhase = phaseMat(freqIdx,:);
            fftStruct.file(fidx).targetMag = magMat(freqIdx,:);
            fftStruct.file(fidx).freqsAtMaxMag = maxFreqs;
            fftStruct.file(fidx).fftMat = fftMat;
            fftStruct.file(fidx).traces = traces;
            fftStruct.file(fidx).sliceIdxs = sliceIdxs;
            fftStruct.file(fidx).targetFreq = targetFreq;
            fftStruct.file(fidx).freqs = freqs;
            fftStruct.file(fidx).targetFreqIdx = freqIdx;
            fftStruct.file(fidx).magMat = magMat;
            fftStruct.file(fidx).ratioMat = ratioMat;
            fftStruct.file(fidx).phaseMat = phaseMat;

            if inferred
                fftStruct.file(fidx).targetPhaseNMF = phaseMatNMF(freqIdx,:);
                fftStruct.file(fidx).targetMagNMF = magMatNMF(freqIdx,:);
                fftStruct.file(fidx).freqsAtMaxMagNMF = maxFreqsNMF;
                fftStruct.file(fidx).fftMatNMF = fftMatNMF;
                fftStruct.file(fidx).tracesNMF = tracesNMF; %detrendedTraceMatNMF;
                fftStruct.file(fidx).magMatNMF = magMatNMF;
                fftStruct.file(fidx).ratioMatNMF = ratioMatNMF;
                fftStruct.file(fidx).phaseMatNMF = phaseMatNMF;

                fftStruct.file(fidx).targetPhaseNMFoutput = phaseMatNMFoutput(freqIdx,:);
                fftStruct.file(fidx).targetMagNMFoutput = magMatNMFoutput(freqIdx,:);
                fftStruct.file(fidx).freqsAtMaxMagNMFoutput = maxFreqsNMFoutput;
                fftStruct.file(fidx).fftMatNMFoutput = fftMatNMFoutput;
                fftStruct.file(fidx).tracesNMFoutput = tracesNMFoutput; %detrendedTraceMatNMF;
                fftStruct.file(fidx).magMatNMFoutput = magMatNMFoutput;
                fftStruct.file(fidx).ratioMatNMFoutput = ratioMatNMFoutput;
                fftStruct.file(fidx).phaseMatNMFoutput = phaseMatNMFoutput;
            end



            % Get MAPS for current slice: ---------------------------------
            tic();
            switch D.roiType
                case 'pixels'
                    phaseMap = reshape(phaseMat(freqIdx,:), [d1, d2]);
                    magMap = reshape(magMat(freqIdx,:), [d1, d2]);
                    ratioMap = reshape(ratioMat, [d1, d2]);
                    phasesAtMaxMag = arrayfun(@(i) freqs(magMat(:,i)==max(magMat(:,i))), 1:nrois);
                    phaseMaxMag = reshape(phasesAtMaxMag, [d1, d2]);
    %             case '3Dcnmf'
    %                 phaseMap = assignRoiMap3D(maskcell, centers, nslices, blankMap, meanDfs);
    %                 
                otherwise
                    phaseMap = assignRoiMap(maskcell, phaseMap, phaseMat, freqIdx);
                    magMap = assignRoiMap(maskcell, magMap, magMat, freqIdx);
                    ratioMap = assignRoiMap(maskcell, ratioMap, ratioMat);
                    phaseMaxMag = assignRoiMap(maskcell, phaseMat, magMat, [], freqs);
            end

    %         % --
            if inferred
                %display('hi')
                phaseMapNMF = assignRoiMap(maskcell, phaseMapNMF, phaseMatNMF, freqIdx);
                magMapNMF = assignRoiMap(maskcell, magMapNMF, magMatNMF, freqIdx);
                ratioMapNMF = assignRoiMap(maskcell, ratioMapNMF, ratioMatNMF);
                phaseMaxMagNMF = assignRoiMap(maskcell, phaseMatNMF, magMatNMF, [], freqs);

                phaseMapNMFoutput = assignRoiMap(maskcell, phaseMapNMFoutput, phaseMatNMFoutput, freqIdx);
                magMapNMFoutput = assignRoiMap(maskcell, magMapNMFoutput, magMatNMFoutput, freqIdx);
                ratioMapNMFoutput = assignRoiMap(maskcell, ratioMapNMFoutput, ratioMatNMFoutput);
                phaseMaxMagNMFoutput = assignRoiMap(maskcell, phaseMatNMFoutput, magMatNMFoutput, [], freqs);


            end
    %         % ---

            toc();

            maps.file(fidx).magnitude = magMap;
            maps.file(fidx).phase = phaseMap;
            maps.file(fidx).phasemax = phaseMaxMag;
            maps.file(fidx).ratio = ratioMap;
            maps.file(fidx).avgY = avgY;

            % ---
            if inferred
                maps.file(fidx).magnitudeNMF = magMapNMF;
                maps.file(fidx).phaseNMF = phaseMapNMF;
                maps.file(fidx).phasemaxNMF = phaseMaxMagNMF;
                maps.file(fidx).ratioNMF = ratioMapNMF;

                maps.file(fidx).magnitudeNMFoutput = magMapNMFoutput;
                maps.file(fidx).phaseNMFoutput = phaseMapNMFoutput;
                maps.file(fidx).phasemaxNMFoutput = phaseMaxMagNMFoutput;
                maps.file(fidx).ratioNMFoutput = ratioMapNMFoutput;


            end
            % ---

            % --- Need to reshape into 2d image if using pixels:
    %         if strcmp(D.roiType, 'pixels')
    %             mapTypes = fieldnames(maps);
    %             for map=1:length(mapTypes)
    %                 currMap = maps.(mapTypes{map});
    %                 currMap = reshape(currMap, [d1, d2, size(currMap,3)]);
    %                 maps.(mapTypes{map}) = currMap;
    %             end
    %         end
            % --------------------------------------------------------------

        end
    
    end
    % Save maps for current slice:

    mapStructName = sprintf('maps_Slice%02d', D.slices(sidx));
    save_struct(D.outputDir, mapStructName, maps);

    fftStructName = sprintf('fft_Slice%02d', D.slices(sidx));
    save_struct(D.outputDir, fftStructName, fftStruct);

    %M.file(fidx) = maps;

    fprintf('Finished FFT analysis for Slice %02d.\n', D.slices(sidx));

    mapStructNames{sidx} = mapStructName;
    fftStructNames{sidx} = fftStructName;
    
end

clear fftStruct maps T

fprintf('TOTAL TIME ELAPSED:\n');
toc(fstart);

D.mapStructNames = mapStructNames;
D.fftStructNames = fftStructNames;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');


%% Get dF/F maps:
% 
% meta = load(D.metaPath);
% 
% minDf = 20;
% 
% fftNames = dir(fullfile(outputDir, 'vecfft_*'));
% fftNames = {fftNames(:).name}';
% 
% dfStruct = struct();
% for sidx = 1:length(slicesToUse)
%     currSlice = slicesToUse(sidx);
%     
%     M = load(maskPaths{sidx});
%     maskcell = M.maskcell;
%     clear M;
%     
%     traceStruct = load(fullfile(tracesPath, traceNames{sidx}));
%     fftName = sprintf('vecfft_Slice%02d.mat', currSlice);
%     fftStruct = load(fullfile(outputDir, fftName));
%     %F = load(fullfile(outputDir, fftNames{sidx}));
%     
%     meanMap = zeros(d1, d2, 1);
%     maxMap = zeros(d1, d2, 1);
%     
%     for fidx=1:length(fftStruct.file)
%         activeRois = [];
%         nRois = length(maskcell);
%         
%         traces = traceStruct.traces.file{fidx};
%         raw = fftStruct.file(fidx).trimmedRawMat;
%         filtered = fftStruct.file(fidx).traceMat;
%         adjusted = filtered + mean(raw,3);
%         %dfFunc = @(x) (x-mean(x))./mean(x);
%         %dfMat = cell2mat(arrayfun(@(i) dfFunc(adjusted(i, :)), 1:size(adjusted, 1), 'UniformOutput', false)');
%         dfMat = arrayfun(@(i) extractDfTrace(adjusted(i, :)), 1:size(adjusted, 1), 'UniformOutput', false);
%         dfMat = cat(1, dfMat{1:end})*100;
%         
%         meanDfs = mean(dfMat,2);
%         maxDfs = max(dfMat, [], 2);
%         activeRois = find(maxDfs >= minDf);
%         fprintf('Found %i ROIs with dF/F > %02.f%%.\n', length(activeRois), minDf);
%         
%         meanMap = assignRoiMap(maskcell, meanMap, meanDfs);
%         maxMap = assignRoiMap(maskcell, maxMap, maxDfs);
%         
%         %meanMap(masks(:,:,1:nRois)==1) = mean(dF,2);
%         %maxMap(masks(:,:,1:nRois)==1) = max(dF,2);
%         
%         dfStruct.file(fidx).meanMap = meanMap;
%         dfStruct.file(fidx).maxMap = maxMap;
%         dfStruct.file(fidx).dfMat = dfMat;
%         dfStruct.file(fidx).activeRois = activeRois;
%         dfStruct.file(fidx).minDf = minDf;
%         dfStruct.file(fidx).maxDfs = maxDfs;
%         
%     end
%     
%     dfName = sprintf('vecdf_Slice%02d', currSlice);
%     save_struct(outputDir, dfName, dfStruct);
% 
% end
