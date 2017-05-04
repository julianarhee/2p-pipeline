function meta = createMetaStruct(D, varargin)

% Combine meta info from acquisition (SI) and experiment (MW/Python).
% Outputs 1 struct that contains info about current acquisition.
% Struct fields include "mw" and "si" relevant info that is parsed by TIFF
% datafile (a single .tiff saved in ScanImage).
interpolateSI = true;

nvargin = length(varargin);
switch nvargin
    case 0
        addOffset = true;
    case 1
        addOffset = false;
end

sourceDir = D.sourceDir;
acquisitionName = D.acquisitionName;
%nTiffs = D.nTiffs;
channelIdx = D.channelIdx;
tiffSource = D.tiffSource;

if isfield(D, 'extraTiffsExcluded')
    nTiffCorrection = length(D.extraTiffsExcluded);
else
    nTiffCorrection = 0;
end
siStruct = get_si_info(D);

% Only grab relevant info for current experiment to create meta, since we
% analyze by condition:
if D.nExperiments > 1
    % Reconstruct siStruct to only include relevant TIFFs:
    tmpfidxs = arrayfun(@(i) strfind(siStruct.SI.file(i).rawTiffPath, D.experiment), 1:length(siStruct.SI.file), 'UniformOutput', 0);
    fidxs = arrayfun(@(i) ~isempty(tmpfidxs{i}), 1:length(tmpfidxs));
    siStruct.acquisitionName = D.experiment;
    siStruct.nTiffs = length(cell2mat(tmpfidxs));
    siStruct.SI.file(1:siStruct.nTiffs) = siStruct.SI.file(fidxs);
    crossref = true;
else
    crossref = false;
end
nTiffs = siStruct.nTiffs;


mwStruct = get_mw_info(sourceDir, nTiffs, nTiffCorrection, crossref);

% Get paths to CORRECTED tiffs:
channelDir = sprintf('Channel%02d', channelIdx);
tmpTiffs = dir(fullfile(sourceDir, tiffSource, channelDir));
tmpTiffs = tmpTiffs(arrayfun(@(x) ~strcmp(x.name(1),'.'), tmpTiffs));
tiffDirs = {tmpTiffs(:).name}';
nTiffFiles = length(tiffDirs);
fprintf('Found %i TIFF stacks for current acquisition analysis.\n', nTiffFiles);
% if nTiffsTmp ~= nTiffs
%     fprintf('Check SI meta file at: %s\n', siStruct.SI.file(1).tiffPath);
%     fprintf('Meta: %i | Found TIFFS in dir: %i.\n', nTiffs, nTiffsTmp);
% end
tiffPaths = cell(1,length(tiffDirs));
for tidx=1:nTiffFiles
    tiffPaths{tidx} = fullfile(sourceDir, tiffSource, channelDir, tiffDirs{tidx});
end


meta = struct();
for fidx=1:nTiffs
    
    % Determine if there is a single experiment file for multiple TIFFs
    % ("multi") or an independent experiment file for each TIFF ("indie"):
    % ---------------------------------------------------------------------
    if strcmp(mwStruct.mw2si, 'multi')
        mw = mwStruct.MW.file(1);
    elseif strcmp(mwStruct.mw2si, 'indie')
        mw = mwStruct.MW.file(fidx);
    end
    si = siStruct.SI.file(fidx);
    
    condTypes = mw.condTypes;
    runOrder = mw.runOrder;
    currMWfidx = mw.MWfidx + fidx - 1;

    
    % Grab the correct MW run based on TIFF order number:
    % ---------------------------------------------------------------------
    for orderIdx=1:length(fieldnames(runOrder))
        if runOrder.(mw.runNames{orderIdx}) == currMWfidx
            currRunName = mw.runNames{orderIdx};
        end
    end
    
    meta.file(fidx).mw.runName = currRunName;
    meta.file(fidx).mw.MWfidx = currMWfidx;
    meta.file(fidx).mw.orderNum = runOrder.(currRunName);
    
    
    % Get timing info for the TIFF volume of current File:
    % ---------------------------------------------------------------------
%     if addOffset    
%         mwSec = (double(mw.pymat.(currRunName).time) - double(mw.pymat.(currRunName).offset)) / 1E6;
%     else
%         mwSec = (double(mw.pymat.(currRunName).time) - double(mw.pymat.(currRunName).time(1))) / 1E6;
%     end
    mwSec = (double(mw.pymat.(currRunName).time) - double(mw.pymat.(currRunName).time(1))) / 1E6;
    if addOffset
        currOffset = double(mw.pymat.(currRunName).offset) / 1E6;
        fprintf('Adding offfset %0.2f ms from trigger receive time.\n', currOffset)
        mwSec = mwSec + currOffset;
    end
    stimStarts = mw.pymat.(currRunName).idxs + 1; % Get indices of cycle (i.e., "trial") starts
    mwDur = double(mw.pymat.(currRunName).MWdur); %/ 1E6; %mwDur = (double(mw.file(fidx).pymat.triggers(currMWfidx,2)) - double(mw.file(fidx).pymat.triggers(currMWfidx,1))) / 1E6;
    
    nFramesPerVolume = si.nFramesPerVolume;
    nTotalFrames = si.nTotalFrames;
    if isfield(mw, 'ardPath')
        fprintf('Using ARD frame trigger times...\n');
        ardSec = (double(mw.pymat.(currRunName).tframe) - double(mw.pymat.(currRunName).tframe(1))) / 1E6;

        % Check for missing SI triggers on ard/mw side:
        if D.metaonly || (length(ardSec) < nTotalFrames)
            fprintf('Missing %i of expected %i frame timestamps.\n', (nTotalFrames - length(ardSec)), nTotalFrames);
            if interpolateSI
                fprintf('Interpolating...\n');
                tFramesExpected = linspace(ardSec(1), ardSec(end), nTotalFrames);
                ardSecInterp = interp1(ardSec, ardSec, tFramesExpected);
                siSec = ardSecInterp;
            end
        else
            siSec = ardSec;
        end
        siDur = double(mw.pymat.(currRunName).SIdur); %/ 1E6;
    else
        % Can either just use SI's time stamps, or interpolate from MW
        % duration...
        if length(si.siFrameTimes)>1
            siSec = si.siFrameTimes; % This is already in SECS.
            %siSec = linspace(0, mwDur, nTotalFrames);
            siDur = (siSec(end) - siSec(1));
        else
            % Use MW trigger times:
            siDur = (double(mw.pymat.(currRunName).MWtriggertimes(2)) - double(mw.pymat.(currRunName).MWtriggertimes(1)))/1E6;
            siSec = linspace(0, siDur, nTotalFrames);
        end
    end
    %siSec = (double(siTimes) - double(siTimes(1)))/ 1E6;

    
    meta.file(fidx).mw.mwSec = mwSec;
    meta.file(fidx).mw.stimStarts = stimStarts;
    meta.file(fidx).mw.mwDur = mwDur;
    meta.file(fidx).mw.siSec = siSec;
    meta.file(fidx).mw.siDur = siDur;
    fprintf('Tiff %i dur is: %02.2d\n', fidx, siSec(end));
    
%     if exist('siFrameTimes')
%         siSec = siFrameTimes;
%     else
%         if isfield(mw.pymat, 'ard_file_durs')
%             siDur = double(mw.pymat.ard_file_durs(currMWfidx));
%             %if sample_us==1 % Each frame has a t-stamped frame-onset (only true if ARD sampling every 200us, isntead of standard 1ms)
%             siSec = (double(mw.pymat.frame_onset_times{currMWfidx}) - double(mw.pymat.frame_onset_times{currMWfidx}(1))) / 1E6; 
%             if length(si_sec) < nTotalFrames % there are missed frame triggers
%                 siSec = linspace(0, siDur, nTotalFrames);
%                 % This is pretty accurate.. only off by ~ 3ms compared to SI's
%                 % trigger times.
%             end
%         else
%             siSec = linspace(0, mwDur, nTotalFrames);
%         end
%     end
    
    
    if strcmp(mwStruct.stimType, 'bar')
        trimLong = 1;
        nCycles = double(mw.info.ncycles);
        targetFreq = mw.info.target_freq;
        %siFrameRate = 1/median(diff(siSec));
        %siVolumeRate = round(siFrameRate/nFramesPerVolume, 2); % 5.58%4.11 %4.26 %5.58
        nTrueFrames = ceil((1/targetFreq)*nCycles*si.siVolumeRate);
        meta.file(fidx).mw.targetFreq = targetFreq;
        meta.file(fidx).mw.nTrueFrames = nTrueFrames;
        
    else
        nCycles = length(stimStarts);
    end
    

    meta.file(fidx).mw.mwPath = mw.mwPath;
    meta.file(fidx).mw.nCycles = nCycles;
    meta.file(fidx).mw.pymat = mw.pymat;
    
    meta.file(fidx).si = si;
    meta.file(fidx).si.tiffPath = tiffPaths{fidx};

    
end

meta.sourceDir = sourceDir;
meta.acquisitionName = siStruct.acquisitionName; %acquisitionName;

meta.tiffPaths = tiffPaths;
meta.nTiffs = nTiffs;
meta.nChannels = siStruct.nChannels;

meta.nMW = mwStruct.mw2si;
meta.stimType = mwStruct.stimType;
condtypes = cell(1,length(condTypes));
if ~strcmp(meta.stimType, 'bar')
    for c=1:length(condTypes)
        condtypes{c} = condTypes{c};
        %condtypes{c} = ['stim' num2str(str2num(condTypes{c})+1)];
    end
else    
    condtypes = mw.condTypes;
%     else
%     for c=1:length(condTypes)
%         condtypes{c} = ['stim' num2str(str2num(condTypes{c})+1)];
%     end
%     meta.condTypes = condtypes;
end
meta.condTypes = condtypes;


% TOD:  fix this so that runmeta info includes all nec. stuff
% TODO2:  fix so that sessionmeta is parent of ALL runs.
% New stuff for DB: -------------------
meta.volumeRate = si.file(1).siVolumeRate;
meta.volumeSizePixels = [si.file(1).frameWidth, si.file(1).frameHeight, si.file(1).nSlices];
if D.tefo
    meta.volumesize = [500, 500, meta.volumeSizePixels*10];
else
    meta.volumesize = [500, 1000, meta.volumeSizePixels*10];
end
meta.tefo = D.tefo;
% -----------------------------------


% Save META struct:
% ---------------------------------------------------------------------
metaStructName = char(sprintf('meta_%s.mat', siStruct.acquisitionName)); %acquisitionName));
metaStructPath = fullfile(sourceDir, 'analysis', 'meta');
meta.metaPath = fullfile(metaStructPath, metaStructName);
if ~exist(metaStructPath)
    mkdir(metaStructPath)
end

save(fullfile(metaStructPath, metaStructName), '-struct', 'meta');

end