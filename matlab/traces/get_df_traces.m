function get_df_traces(A, varargin)


file_type = 'visible';

ntiffs = A.ntiffs;

switch length(varargin)
    case 0
        df_min = 20;
    case 1
        df_min = varargin{1};
end

DF = struct();
for sidx = 1:length(A.slices)
   
    sl = A.slices(sidx);
    fprintf('Processing SLICE %i...\n', sl);
    
%     if ~strcmp(D.roiType, 'pixels')
%         masks = load(D.maskPaths{sidx});
%     end

    tracestruct = load(fullfile(A.trace_dir, A.trace_structs{sidx}));
        
    for fidx=1:ntiffs
        maskcell = tracestruct.file(fidx).maskcell;

        activeRois = [];
        if strcmp(file_type, 'visible')
            file_dir = sprintf('File%03d_visible', fidx);
        else
            file_dir = sprintf('File%03d', fidx);
        end
        avg_slice_dir = fullfile(A.tiff_source, 'Averaged_Slices', sprintf('Channel%02d', A.signal_channel), file_dir);
        
        slice_files = dir(fullfile(avg_slice_dir, sprintf('*_Slice%02d*', sl)));
        slice_file = slice_files(1).name 
        avgY = tiffRead(fullfile(avg_slice_dir, slice_file));
        traces = tracestruct.file(fidx).tracematDC; 
        % --> This is already corrected with DC -- do the following to get back
        % DC offset removed:  traceMat = bsxfun(@plus, DCs, traceMat);
        [d1,d2] = size(avgY);
        [nframes, nrois] = size(traces);
        fprintf('N frames: %i, N rois: %i\n', nframes, nrois);
                
%         switch D.roiType
%             case 'manual2D'
%                 [d1,d2] = size(avgY);
%                 [nframes, nrois] = size(traces);
%             case 'roiMap'
%                 [d1,d2] = size(avgY);
%                 [nframes, nrois] = size(traces);
%             case 'condition'
%                 [d1,d2] = size(avgY);
%                 [nframes, nrois] = size(traces);
%             case 'pixels'
%                 %[d1,d2,tpoints] = size(T.traces.file{fidx});
%                 [d1, d2] = size(avgY);
%                 nframes = size(traces,1);
%                 nrois = d1*d2;
%             case 'cnmf'
%                 [d1,d2] = size(avgY);
%                 [nframes, nrois] = size(traces);
%         end
        meanMap = zeros(d1, d2, 1);
        maxMap = zeros(d1, d2, 1);
            
        %traces = tracestruct.traces.file{fidx};
        %raw = fftStruct.file(fidx).trimmedRawMat;
        %filtered = fftStruct.file(fidx).traceMat;
        %adjusted = filtered + mean(raw,3);
        %traces = tracestruct.traceMat.file{fidx}; 

        
        %dfFunc = @(x) (x-mean(x))./mean(x);
        %dfMat = cell2mat(arrayfun(@(i) dfFunc(adjusted(i, :)), 1:size(adjusted, 1), 'UniformOutput', false)');
%         dfMat = arrayfun(@(i) extract_df(traces(i, :)), 1:size(traces, 1), 'UniformOutput', false);
        dfMat = arrayfun(@(i) extract_df(traces(:,i)), 1:nrois, 'UniformOutput', false);
        dfMat = cat(2, dfMat{1:end})*100;
        
        meanDfs = mean(dfMat,1);
        maxDfs = max(dfMat);
        
        % Get rid of ridiculous values, prob edge effects:
        maxDfs(abs(maxDfs)>500) = NaN;
        activeRois = find(maxDfs >= df_min);
        fprintf('Found %i of %i ROIs with dF/F > %02.f%%.\n', length(activeRois), nrois, df_min);
        
%         if strcmp(D.roiType, 'pixels')
%             % Just need to reshape into 2d image if using pixels:
%             meanMap = reshape(meanDfs, [d1, d2]);
%             maxMap = reshape(maxDfs, [d1, d2]);
%         else
          meanMap = assign_roimap(maskcell, meanMap, meanDfs);
          maxMap = assign_roimap(maskcell, maxMap, maxDfs);
%         end
        
        % --- Need to reshape into 2d image if using pixels:
%         if strcmp(D.roiType, 'pixels')
%             meanMap = reshape(meanMap, [d1, d2, size(meanMap,3)]);
%             maxMap = reshape(maxMap, [d1, d2, size(maxMap,3)]);
%         end

        % ----------------------------------------------------
        
        
        DF.slice(sl).file(fidx).meanMap = meanMap;
        DF.slice(sl).file(fidx).maxMap = maxMap;
        DF.slice(sl).file(fidx).dfMat = dfMat;
        DF.slice(sl).file(fidx).activeRois = activeRois;
        DF.slice(sl).file(fidx).df_min = df_min;
        DF.slice(sl).file(fidx).maxDfs = maxDfs;
        
    end
    
%     dfName = sprintf('df_Slice%02d', sl);
%     save_struct(D.outputDir, dfName, dfstruct);
%     
%     D.dfStructName = dfName;
%     save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');
%     
    %DF.slice(sl) = dfstruct;
    
    
end

dfName = sprintf('dfstruct.mat');
save_struct(A.trace_dir, dfName, DF);

DF.name = dfName;

% D.dfStructName = dfName;
% save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

   
end
