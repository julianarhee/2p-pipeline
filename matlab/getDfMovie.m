function dfstruct = getDfMovie(D, varargin)

% Get dF/F maps:

meta = load(D.metaPath);
nTiffs = meta.nTiffs;

switch length(varargin)
    case 0
        minDf = 20;
    case 1
        minDf = varargin{1};
end

DF = struct();

%fftNames = dir(fullfile(D.outputDir, fftPrepend));
%fftNames = {fftNames(:).name}';

slicesToUse = D.slices;

dfstruct = struct();
for sidx = 1:length(slicesToUse)
   
    
    currSlice = slicesToUse(sidx);
    
    M = load(D.maskPaths{sidx});
    maskcell = M.maskcell;
    clear M;
    
    tracestruct = load(fullfile(D.tracesPath, D.traceNames{sidx}));
    %fftName = sprintf('%s_Slice%02d.mat', fftPrepend, currSlice);
    %fftStruct = load(fullfile(D.outputDir, fftName));
    %F = load(fullfile(outputDir, fftNames{sidx}));
        
    for fidx=1:nTiffs
        activeRois = [];
        nRois = length(maskcell);
        
        avgY = tracestruct.avgImage.file{fidx};
        adjustTraces = tracestruct.traceMat.file{fidx}; 
        % --> This is already corrected with DC -- do the following to get back
        % DC offset removed:  traceMat = bsxfun(@plus, DCs, traceMat);
        
        switch D.roiType
            case 'create_rois'
                [d1,d2] = size(avgY);
                [nrois, tpoints] = size(adjustTraces);
            case 'condition'
                [d1,d2] = size(avgY);
                [nrois, tpoints] = size(adjustTraces);
            case 'pixels'
                %[d1,d2,tpoints] = size(T.traces.file{fidx});
                [d1, d2] = size(avgY);
                tpoints = size(adjustTraces,2);
                nrois = d1*d2;
        end
        meanMap = zeros(d1, d2, 1);
        maxMap = zeros(d1, d2, 1);
            
        %traces = tracestruct.traces.file{fidx};
        %raw = fftStruct.file(fidx).trimmedRawMat;
        %filtered = fftStruct.file(fidx).traceMat;
        %adjusted = filtered + mean(raw,3);
        %adjustTraces = tracestruct.traceMat.file{fidx}; 

        
        %dfFunc = @(x) (x-mean(x))./mean(x);
        %dfMat = cell2mat(arrayfun(@(i) dfFunc(adjusted(i, :)), 1:size(adjusted, 1), 'UniformOutput', false)');
        dfMat = arrayfun(@(i) extractDfTrace(adjustTraces(i, :)), 1:size(adjustTraces, 1), 'UniformOutput', false);
        dfMat = cat(1, dfMat{1:end})*100;
        
        meanDfs = mean(dfMat,2);
        maxDfs = max(dfMat, [], 2);
        activeRois = find(maxDfs >= minDf);
        fprintf('Found %i of %i ROIs with dF/F > %02.f%%.\n', length(activeRois), size(dfMat,1), minDf);
        
        meanMap = assignRoiMap(maskcell, meanMap, meanDfs);
        maxMap = assignRoiMap(maskcell, maxMap, maxDfs);
        
        %meanMap(masks(:,:,1:nRois)==1) = mean(dF,2);
        %maxMap(masks(:,:,1:nRois)==1) = max(dF,2);
        
        DF.slice(currSlice).file(fidx).meanMap = meanMap;
        DF.slice(currSlice).file(fidx).maxMap = maxMap;
        DF.slice(currSlice).file(fidx).dfMat = dfMat;
        DF.slice(currSlice).file(fidx).activeRois = activeRois;
        DF.slice(currSlice).file(fidx).minDf = minDf;
        DF.slice(currSlice).file(fidx).maxDfs = maxDfs;
        
    end
    
%     dfName = sprintf('df_Slice%02d', currSlice);
%     save_struct(D.outputDir, dfName, dfstruct);
%     
%     D.dfStructName = dfName;
%     save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
%     
    %DF.slice(currSlice) = dfstruct;
    
    
end

dfName = sprintf('dfstruct.mat');
save_struct(D.outputDir, dfName, DF);

D.dfStructName = dfName;
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

   
end