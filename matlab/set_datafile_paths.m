function D = set_datafile_paths(D, dsoptions)

% --------------------------------------------------------------------
% Explicitly set each type of TIFF source and/or acquisition.
% --------------------------------------------------------------------
% Assumptions:

% D.sourceDir :  Parent directory containing current session's RUN.
% - contains raw .tif's (acquisition output of SI) 
% - may contain child dir 'DATA' if preprocessed to remove artifacts,
% namely -- python createsubstacks.py 

% D.dataDir :  This can be the same as D.sourceDir (if no preprocessing),
% or will be child of D.sourceDir

% D.mempath :  For large tiffs, will be better to work with memmapped
% files. This is the path into which memmap3D() will save those .mat's

% D.averagePath :  For noisy data, sometimes want to combine raw tiffs
% across repetitions. In this case, want to specify the path to store,
% or path that is already storing, the averaged (memmapped) files.

% D.tiffSource :  Ff motion-corrected, this will be saved to 'Corrected'
% which is child dir D.dataDir. Otherwise, will just be in 'Parsed'. 
% TODO:  this may not be necessary to specify (or even do)...
% --------------------------------------------------------------------


if isempty(dsoptions.datapath)
    D.processedtiffs = false;
else
    D.processedtiffs = true;
end
D.dataDir = fullfile(D.sourceDir, dsoptions.datapath);

D.mempath = fullfile(D.dataDir, 'memfiles');
if ~exist(D.mempath)
    mkdir(D.mempath)
end

if D.average
    D.matchtrials = dsoptions.matchedtiffs; %matchtrials;
    D.averagePath = fullfile(D.mempath, 'averaged');
    if ~exist(D.averagePath, 'dir')
        mkdir(D.averagePath)
    end
end

D.sliceimagepath = fullfile(D.dataDir, 'average_slices');
if ~exist(D.sliceimagepath, 'dir')
    mkdir(D.sliceimagepath)
end

%% Get SI volume info:

% metaInfo = 'SI';
% 
% corrected = true;
if dsoptions.corrected
    D.tiffSource = fullfile(D.dataDir, 'Corrected');
else
    D.tiffSource = fullfile(D.dataDir, 'Parsed');
    D.metaonly = dsoptions.metaonly;
    if D.metaonly
        D.nTiffs = dsoptions.nmetatiffs;
    end
end

% Check if more than one "experiment" in current motion-correction
% acquisition (i.e., multiple conditions in session corrected to a single
% reference run of some other condition):
%tmpDirs = dir(fullfile(D.so;urceDir, D.tiffSource));
tmpDirs = dir(D.tiffSource)
tmpDirs = tmpDirs(arrayfun(@(x) x.name(1), tmpDirs) ~= '.');
tmpDirs = tmpDirs(find(~cell2mat(strfind(arrayfun(@(x) x.name, tmpDirs, 'UniformOutput', 0),'Channel'))));

if numel(tmpDirs) > 1
    D.nExperiments = length(tmpDirs);
    D.experimentNames = arrayfun(@(x) tmpDirs(x).name, 1:length(tmpDirs), 'UniformOutput', 0);
    fprintf('More than 1 condition corrected with current refernce:\n');
    for eidx=1:length(D.experimentNames)
        fprintf('Idx: %i, Name: %s\n', eidx, D.experimentNames{eidx});
    end
    selectedExperiment = input('Select idx of condition to analyze:\n');
    D.tiffSource = fullfile(D.tiffSource, D.experimentNames{selectedExperiment});
    D.experiment = D.experimentNames{selectedExperiment};
else
    D.nExperiments = 1;
    D.experimentNames = D.acquisitionName;
    D.experiment = D.acquisitionName;
end
    
save(fullfile(D.datastructPath, D.name), '-append', '-struct', 'D');

fprintf('Got SI file info.\n');

end
