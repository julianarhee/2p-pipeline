function M = get_mw_info(sourceDir, nTiffs, nTiffCorrection, crossref)

M = struct();
mwStruct = struct();

mwPath = fullfile(sourceDir, 'mw_data');

fileNames = dir(fullfile(mwPath, '*.mat'));
if isempty(fileNames)
    
    fprintf('Failed to find corresponding MW-parsed .mat file for current acquisition %s.\n', mwPath);
    
    if exist(mwPath,'dir')
        fprintf('Found MW path at %s, but .mwk file is not yet parsed.\n', mwPath);
    end
    mw2si = 'unknown';
else
    fileNames = {fileNames(:).name}';
    if crossref
        tmpmats = {};
        for f=1:length(fileNames)
            mplaces = find(fileNames{f}=='_');
            tmpmats{end+1} = fileNames{f}(1:mplaces(end)-1);
        end
        matnames = unique(tmpmats);
        fprintf('More than 1 .mat type found. Looks like multiple conditions are corrected to same reference.\n');
        for f=1:length(matnames)
            fprintf('Idx: %i, MAT: %s\n', f, matnames{f})
        end
        matroot = matnames{input('Select IDX of curr analysis:\n')};
        tmpfidxs = cellfun(@(n) strfind(n, matroot), fileNames, 'UniformOutput', 0);
        fidxs = arrayfun(@(i) ~isempty(tmpfidxs{i}), 1:length(tmpfidxs));
        fileNames = fileNames(fidxs);
    end
    
    fprintf('Found %i MW files and %i TIFFs.\n', length(fileNames), nTiffs);
    pymatName = fileNames{1};
    pymat = load(fullfile(mwPath, pymatName));
    nRuns = size(pymat.runs,1); %length(pymat.runs); % Just check length of a var to see how many "runs" MW-parsing detected.
   
    fprintf('---------------------------------------------------------\n');
    if length(fileNames) == nTiffs+nTiffCorrection 
        % There is 1 .mwk file per .tiff file, including the excluded
        % TIFFs.
        fprintf('Looks like indie runs.\n');
        mw2si = 'indie';
        MWfidx = 1;
    elseif length(fileNames)==1
        % There is only a SINGLE .mwk file for all TIFFs, and the number of
        % chunks or runs in the MWK file corresponds to the # of TIFFs
        % included in analysis, plus excluded TIFFs.
        fprintf('Multiple SI runs found in current MW file %s.\n', pymatName);
        fprintf('MW detected %i runs, and there are %i TIFFs.\n', nRuns, nTiffs);
        fprintf('---------------------------------------------------------\n');
        if nRuns==(nTiffs+nTiffCorrection) % Likely, discarded/ignored TIFF files for given MW session
            MWfidx = 1;
            mw2si = 'multi';
            fprintf('Found correct # of runs in MW session to match # TIFFs.\n');
            fprintf('Setting mw-fix to 1.\n');
        elseif nRuns > (nTiffs+nTiffCorrection)            
            MWfidx = (nRuns - (nTiffs+nTiffCorrection)) + 1; % EX: TIFF file1 is really corresponding to MW 'run' 4 if ignoring first 3 TIFF files.
            mw2si = 'multi';
            fprintf('Looks like %i TIFF files should be ignored.\n', (nRuns-nTiffs));
            fprintf('Also excluding %i TIFF files from the end...\n', nTiffCorrection);
            fprintf('Setting MW file start index to %i.\n', MWfidx);
        else
            fprintf('**Warning** There are fewer MW runs than TIFFs. Is MW file parsed correctly?\n');
            mw2si = 'unknown';
        end
    else               
        fprintf('Mismatch in num of MW files (%i .mwk found) and TIFF files (%i .tif found).\n', length(fileNames), nTiffs);
        mw2si = 'unknown';        
    end
end

for midx=1:length(fileNames)
    pymat = load(fullfile(mwPath, fileNames{midx}));
    if isfield(pymat, 'ard_dfn')
        mwStruct.file(midx).ardPath = pymat.ard_dfn;
    end
    mwStruct.file(midx).mwPath = fullfile(mwPath, fileNames{midx});
    mwStruct.file(midx).MWfidx = MWfidx;
    mwStruct.file(midx).stimType = pymat.stimtype;
    mwStruct.file(midx).runNames = cellstr(pymat.runs);
    if strcmp(pymat.stimtype, 'grating')
        mwStruct.file(midx).condTypes = {};
        for gidx=1:size(pymat.condtypes,1)
            mwStruct.file(midx).condTypes{end+1} = sprintf('stim%i', gidx);
        end
        mwStruct.file(midx).condValues = pymat.condtypes;
        %t = arrayfun(@(gidx) mat2str(tmp(gidx,:)), 1:size(tmp,1), 'UniformOutput', false)
    else
        mwStruct.file(midx).condTypes = cellstr(pymat.condtypes);
    end
    for run=1:length(mwStruct.file(midx).runNames)
        mwStruct.file(midx).pymat.(mwStruct.file(midx).runNames{run}) = pymat.(mwStruct.file(midx).runNames{run});
    end
    if strcmp(pymat.stimtype, 'bar')
        mwStruct.file(midx).info = pymat.info;
    end
end

               
% switch mw2si
%     case 'indie' % Each SI file goes with one MW/ARD file
%         % do stuff
%         for midx=1:length(fileNames)
%             pymat = load(fullfile(mwPath, fileNames{midx}));
%             
%             tmpRunNames = cellstr(pymat.runs);
%             mwStruct.file(midx).mwPath = fullfile(mwPath, fileNames{midx});
%             mwStruct.file(midx).MWfidx = MWfidx;
%             mwStruct.file(midx).stimType = pymat.stimtype;
%             mwStruct.file(midx).runNames = {tmpRunNames{MWfidx:end}};
%             mwStruct.file(midx).condTypes = cellstr(pymat.condtypes);
%             mwStruct.file(midx).pymat = pymat;
%             
% 
%         end
%         
%     case 'multi'
%         % do other stuff
%         pymatName = fileNames{1}; % There should only be 1 mwk file for multiple TIFFs (experiment reps).
%         pymat = load(fullfile(mwPath, pymatName));
%         %runNames = cellstr(pymat.conditions);
%         if strcmp(pymat.stimtype, 'bar')
%             % sthmight be off here, edited..
%             runNames = cellstr(pymat.runs);
%             condTypes = cellstr(pymat.condtypes);
%         elseif strcmp(pymat.stimtype, 'grating')
%             condTypes = cellstr(pymat.condtypes);
%         elseif strcmp(pymat.stimtype, 'image')
%             runNames = cellstr(pymat.runs);
%             condTypes = cellstr(pymat.condtypes);
%         else
%             fprintf('No stimype detected in MW file...\n');
%         end
% 
%     otherwise
%         % fix stuff.
% end


%M.runNames = runNames;
M.fileNames = fileNames;
M.mw2si = mw2si;
M.nRuns = nRuns;
M.nTiffs = nTiffs;
M.stimType = pymat.stimtype;


% Create arbitrary stimtype codes:
% stimTypes = cell(1,length(mwStruct.file(1).condTypes));
% for sidx=1:length(mwStruct.file(1).condTypes)
%     sname = sprintf('code%i', sidx);
%     stimTypes{sidx} = sname;
% end

% Get indices of each run to preserve order when indexing into MW
% file-structs:
for fileIdx=1:length(fileNames)
    
    runOrder = struct();
    currRunNames = mwStruct.file(fileIdx).runNames;
    for runIdx=1:length(currRunNames)
        runOrder.(currRunNames{runIdx}) = mwStruct.file(fileIdx).pymat.(currRunNames{runIdx}).ordernum + 1;
    end
    mwStruct.file(fileIdx).runOrder = runOrder;
    
end

%M.stimTypes = stimTypes;
%M.runOrder = runOrder;

M.MW = mwStruct;

%------------------------


% mwStructName = char(sprintf('MW_%s.mat', acquisitionName));
% mwStructPath = fullfile(analysisDir, 'MW');
% if ~exist(mwStructPath)
%     mkdir(mwStructPath)
% end
% 
% save(fullfile(mwStructPath, mwStructName), '-struct', 'M');



end