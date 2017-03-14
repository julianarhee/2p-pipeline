% RETINOTOPY.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

acquisition_info;

% -------------------------------------------------------------------------
% FFT analysis:
% -------------------------------------------------------------------------
% For each analysed slice, do FFT analysis on each file (i.e., run, or
% condition rep).
% For a given set of traces (on a given run) extracted from a slice, use 
% corresponding metaInfo for maps.

for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    
    M = load(maskPaths{sidx});
    masks = M.masks;
    clear M;
        
    T = load(fullfile(tracesDir, traceNames{sidx}));
    
    fftStruct = struct();
    for fidx=1:nTiffs
        
        meta = load(metaPaths{fidx});
        
        traces = T.traces.file{fidx};
        avgY = T.avgImage.file{fidx};
        

        targetFreq = meta.targetFreq;
        ncycles = meta.ncycles;
        nTotalSlices = meta.nFramesPerVolume;
        
        cutEnd=1;
        crop = meta.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
        
        switch roiType
            case 'create_rois'
                [d1,d2] = size(avgY);
                [nrois, tpoints] = size(T.traces.file{fidx});
            case 'pixels'
                %[d1,d2,tpoints] = size(T.traces.file{fidx});
                [d1, d2] = size(avgY);
                tpoints = size(T.traces.file{fidx},3);
        end
        
        % Get phase and magnitude maps:
        phaseMap = zeros(d1, d2, 1);
        magMap = zeros(d1, d2, 1);
        ratioMap = zeros(d1, d2, 1);
        phaseMaxMag = zeros(d1, d2, 1);

        for roi=1:size(traces,1)
            
            if mod(roi,10)==0
                fprintf('Processing roi #%i...\n', roi)
            end
            
            sliceIdxs = currSlice:meta.nFramesPerVolume:meta.nTotalFrames;
            currTrace = traces(roi, :);
            
            % Subtract rolling mean to get rid of drift:
            % -------------------------------------------------------------
%             if check_slice
%                 tmp0(:) = squeeze(currTrace(1:end));
%             end

            Fs = meta.siVolumeRate;
            winsz = round((1/targetFreq)*Fs*2);
           
            if cutEnd==1
                tmp0 = currTrace(1:crop);
            else
                tmp0 = currTrace;
            end
            tmp1 = padarray(tmp0,[0 winsz],tmp0(1),'pre');
            tmp1 = padarray(tmp1,[0 winsz],tmp1(end),'post');
            rollingAvg=nanconv(tmp1,fspecial('average',[1 winsz]),'same');%average
            rollingAvg=rollingAvg(winsz+1:end-winsz);
            trace_y = tmp0 - rollingAvg;
            
            % Do FFT:
            % -------------------------------------------------------------
            N = length(trace_y);
            dt = 1/Fs;
            t = dt*(0:N-1)';
            dF = Fs/N;
            freqs = dF*(0:N/2-1)';
            
            fft_y = fft(trace_y)/N;
            fft_y = fft_y(1:N/2);
            fft_y(2:end) = 2*fft_y(2:end);
            %freqs = Fs*(0:(NFFT/2))/NFFT;
            freqIdx = find(abs((freqs-targetFreq))==min(abs(freqs-targetFreq)));
            %fprintf('Target frequency %02.2f found at idx %i.\n', targetFreq, freqIdx);
            
            magY = abs(fft_y);
            ratioY = magY(freqIdx) / (sum(magY) - magY(freqIdx));
            %phaseY = unwrap(angle(Y));
            phaseY = angle(fft_y);
            %phase_rad = atan2(imag(Y(freqIdx)), real(Y(freqIdx)));
            %phase_deg = phase_rad / pi * 180 + 90;
            
            % Store FFT analysis resuts to struct:
            % -------------------------------------------------------------
            fftStruct.file(fidx).roi(roi).targetPhase = phaseY(freqIdx); % unwrap(angle(Y(freqIdx)));
            fftStruct.file(fidx).roi(roi).targetMag = magY(freqIdx);
            fftStruct.file(fidx).roi(roi).maxFreq = freqs(magY==max(magY(:)));
            fftStruct.file(fidx).roi(roi).fft = fft_y;
            fftStruct.file(fidx).roi(roi).trace = trace_y;
            fftStruct.file(fidx).roi(roi).raw = tmp0;
            fftStruct.file(fidx).roi(roi).slices = sliceIdxs;
            fftStruct.file(fidx).roi(roi).freqs = freqs;
            fftStruct.file(fidx).roi(roi).freqIdx = freqIdx;
            fftStruct.file(fidx).roi(roi).targetFreq = targetFreq;

            phaseMap(masks(:,:,roi)==1) = phaseY(freqIdx);
            magMap(masks(:,:,roi)==1) = magY(freqIdx);
            ratioMap(masks(:,:,roi)==1) = ratioY;
            phaseMaxMag(masks(:,:,roi)==1) = phaseY(magY==max(magY));
            
            maps.file(fidx).magnitude = magMap;
            maps.file(fidx).phase = phaseMap;
            maps.file(fidx).phasemax = phaseMaxMag;
            maps.file(fidx).ratio = ratioMap;
            maps.file(fidx).avgY = avgY;
                            
            % --- Need to reshape into 2d image if using pixels:
            if strcmp(roiType, 'pixels')
                mapTypes = fieldnames(maps);
                for map=1:length(mapTypes)
                    currMap = maps.(mapTypes{map});
                    currMap = reshape(currMap, [d1, d2, size(currMap,3)]);
                    maps.(mapTypes{map}) = currMap;
                end
            end
            % ---
            
        end
        
    end
    
    % Save maps for current slice:
    map_fn = sprintf('maps_Slice%02d', currSlice);
    save_struct(outputDir, map_fn, maps);

    fft_fn = sprintf('fft_Slice%02d', currSlice);
    save_struct(outputDir, fft_fn, fftStruct);

    %M.file(fidx) = maps;
    
    fprintf('Finished FFT analysis for Slice %02d.\n', currSlice);
    
end
clear fftStruct maps T

%% Get dF/F maps:

minDf = 10;

fftNames = dir(fullfile(outputDir, 'fft_*'));
fftNames = {fftNames(:).name}';

dfStruct = struct();
for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    
    M = load(maskPaths{sidx});
    masks = M.masks;
    clear M;
    
    T = load(fullfile(tracesDir, traceNames{sidx}));
    F = load(fullfile(outputDir, fftNames{sidx}));
    
    meanMap = zeros(d1, d2, 1);
    maxMap = zeros(d1, d2, 1);
    
    for fidx=1:length(F.file)
        tmpDf = struct();
        activeRois = [];
        for roi=1:length(F.file(fidx).roi)
            
            if mod(roi, 20)==0
                fprintf('Processing roi #%i...\n', roi);
            end
            meta = load(metaPaths{fidx});
            traces = T.traces.file{fidx};
            
            rawTrace = F.file(fidx).roi(roi).raw;
            filteredTrace = F.file(fidx).roi(roi).trace;
            adjustedTrace = filteredTrace + mean(rawTrace);
            dF = ((adjustedTrace - mean(adjustedTrace))./mean(adjustedTrace)) * 100;
            if max(dF(:)) > minDf
                activeRois = [activeRois roi];
            end
            
            % TO DO:  fix this maps so we plot sth that is actually
            % useful... z-score map maybe?
            
            meanMap(masks(:,:,roi)==1) = mean(dF); 
            maxMap(masks(:,:,roi)==1) = max(dF(:));
            tmpDf.dfTrace{roi} = dF;
            tmpDf.activeRois{roi} = activeRois;
            
        end
        fprintf('Found %i ROIs with dF/F > %0.2f%%.\n', length(activeRois), minDf);
        dfStruct.file(fidx).meanMap = meanMap;
        dfStruct.file(fidx).maxMap = maxMap;
        dfStruct.file(fidx).traces = tmpDf.dfTrace;
        dfStruct.file(fidx).activeRois = tmpDf.activeRois;
        clear dF
    end
    dfName = sprintf('df_Slice%02d', currSlice);
    save_struct(outputDir, dfName, dfStruct);
        
end

%% ROI figures:

alphaVal = 0.5;
for sidx = 1:length(slicesToUse)
    currSlice = slicesToUse(sidx);
    
    M = load(maskPaths{sidx});
    masks = M.masks;
    nROIs = size(masks,3);
    
    clear M;
    
    T = load(fullfile(tracesDir, traceNames{sidx}));
    F = load(fullfile(outputDir, fftNames{sidx}));
    
    meanMap = zeros(d1, d2, 1);
    maxMap = zeros(d1, d2, 1);
    
    for fidx=1:length(F.file)
        
        avgY = T.avgImage.file{1};
        
        RGBimg = zeros([size(avgY),3]);
        RGBimg(:,:,1)=0;
        RGBimg(:,:,2)=mat2gray(avgY); %mat2gray(avgY);
        RGBimg(:,:,3)=0;
        
        for roi=1:nROIs
            RGBimg(:,:,3) = RGBimg(:,:,3)+alphaVal*masks(:,:,roi);
        end
        
        fig=figure();
        imshow(RGBimg);
        title(sprintf('Slice%02d File%03d', currSlice, fidx));
        
        roiFigName = sprintf('ROIs_Slice%02d_File%03d', currSlice, fidx);
        
        figSliceDir = sprintf('Slice%02d', currSlice);
        currFigDir = fullfile(D.figDir, figSliceDir);
        
        save_figures(fig, currFigDir, roiFigName)
            
    end
end

%% PLOTTING.

subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.1], [0.1 0.1], [0.1 0.1]);

cmap = 'hsv';

fftNames = dir(fullfile(outputDir, 'maps_*'));
fftNames = {fftNames(:).name}';
dfNames = dir(fullfile(outputDir, 'df_*'));
dfNames = {dfNames(:).name}';

D.fftNames = fftNames;
D.dfNames = dfNames;
save(fullfile(analysisDir, datastruct), '-append', '-struct', 'D');


% get legends if needed:
if isempty(dir(fullfile(outputDir, '*legends.mat')))
    fprintf('generating legends...\n');
    make_legends(D.fftDir);
end
legendPath = dir(fullfile(outputDir, '*legends.mat'));
legendPath = {legendPath(:).name}';
legends = load(char(fullfile(outputDir, legendPath)));


for sidx=1:length(fftNames)
    
    FFT = load(fullfile(outputDir, fftNames{sidx})); 
    DF = load(fullfile(outputDir, dfNames{sidx}));
    
    currSlice = D.slices(sidx);
    
    for fidx=1:length(FFT.file)
        
        meta = load(metaPaths{fidx});
        currCond = meta.mwRunName;
        currCond = strrep(currCond, '_', '-');
        condTypeParts = strsplit(currCond,'-');
        currCondType = condTypeParts{1};
        
        phaseMap = FFT.file(fidx).phase;
        magMap = FFT.file(fidx).magnitude;
        ratioMap = FFT.file(fidx).ratio;
        phaseMaxMap = FFT.file(fidx).phasemax;
        
        maxDfMap = DF.file(fidx).maxMap;
        
        %%
        fov = repmat(mat2gray(avgY), [1, 1, 3]);
        
        fig = figure();
        slice_no = sprintf('slice%02d', sidx);
        file_no = sprintf('file%03d', fidx);

        % surface ------------------------------
        ax1 = subplot(2,3,1);
        imagesc2(avgY);
        axis('off')
        %hb = colorbar('location','eastoutside');
        colormap(ax1, gray)
        ax1Pos = get(ax1, 'position');
        
        % ratio of mags -------------------------
        ax2 = subplot(2,3,2);
        magOverlay = threshold_map(ratioMap, magMap, 0);
        imagesc(fov);
        hold on;
        imagesc2(magOverlay)
        axis('off')
        colormap(ax2, hot)
        hb = colorbar('location','eastoutside');
        title('mag ratio')
        ax2Pos = get(ax2,'position');
        ax2Pos(3:4) = [ax1Pos(3:4)];
        set(ax2, 'position', ax2Pos);
        
        % dF/F map -----------------------------
        ax3 = subplot(2,3,3);
        dFoverlay = threshold_map(maxDfMap, magMap, 0);
        imagesc(fov);
        hold on;
        imagesc2(dFoverlay);
        axis('off')
        colormap(ax3, hot)
        hb = colorbar('location','eastoutside');
        title('dF/F')
        ax3Pos = get(ax3,'position');
        ax3Pos(3:4) = [ax1Pos(3:4)];
        set(ax3, 'position', ax3Pos);
        
        % phase ----------------------------------
        
        ax4 = subplot(2,3,4);
        threshold = 0.1;
        thresholdPhase = threshold_map(phaseMap, magMap, threshold);
        
        imagesc(fov);
        hold on;
        imagesc2(thresholdPhase);
        title('phase')
        colormap(ax4, cmap)
        caxis([-pi, pi])
        ax4Pos = get(ax4,'position');
        %ax4Pos(3:4) = [ax1Pos(3:4)];
        %set(ax4, 'position', ax4Pos);

        % phase at max mag ---------------------
        ax5 = subplot(2,3,5);

        phaseMaxOverlay = threshold_map(phaseMaxMap, magMap, threshold);
        imagesc(fov);
        hold on;
        imagesc2(phaseMaxOverlay);
        title('phase')
        colormap(ax4, cmap)
        caxis([-pi, pi])
        ax5Pos = get(ax5,'position');
        ax5Pos(3:4) = [ax1Pos(3:4)];
        set(ax5, 'position', ax5Pos);
        
        
        % legend -------------------------------
        ax6 = subplot(2,3,6);
        imagesc2(legends.(currCondType))
        axis('off')
        caxis([-pi, pi])
        colormap(ax6, cmap)
        ax6Pos = get(ax6,'position');
        ax6Pos(3:4) = [ax1Pos(3:4)];
        set(ax6, 'position', ax6Pos);
        
        
        set(fig, 'Position', [995 1046 1411 288], 'Units', 'pixels');
        
        %%
        
        suptitle(sprintf('Slice%02d, Cond: %s', currSlice, currCond))
        
        figName = sprintf('Overview_Slice%02d_File%03d_%s', currSlice, fidx, currCond);
        figSliceDir = sprintf('Slice%02d', currSlice);
        currFigDir = fullfile(D.figDir, figSliceDir);
  
        save_figures(fig, currFigDir, figName);

    end
    
end


%%

