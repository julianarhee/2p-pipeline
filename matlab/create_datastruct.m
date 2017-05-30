function D = create_datastruct(dsoptions)

% --------------------------------------------------------------------
% Create datastruct and save to file using input parameters.
% --------------------------------------------------------------------

sourceDir = fullfile(dsoptions.source, dsoptions.session, dsoptions.run); 
didx = dsoptions.datastruct; 
acquisitionName = dsoptions.acquisition; 
corrected = dsoptions.corrected; 
 
analysisDir = fullfile(sourceDir, 'analysis'); 
if ~exist(analysisDir, 'dir') 
    mkdir(analysisDir); 
end 

 
datastruct = sprintf('datastruct_%03d', didx); 
dstructPath = fullfile(analysisDir, datastruct); 
existingAnalyses = dir(analysisDir); 
existingAnalyses = {existingAnalyses(:).name}'; 
 
if ~exist(dstructPath) 
    mkdir(dstructPath) 
    D = struct(); 
else 
    fprintf('*********************************************************\n'); 
    fprintf('WARNING:  Specified datastruct -- %s -- exists. Overwrite?\n', datastruct); 
    uinput = input('Press Y/n to overwrite or create new: \n', 's'); 
    if strcmp(uinput, 'Y') 
        D = struct(); 
        fprintf('New datastruct created: %s.\n', datastruct); 
        fprintf('Not yet saved. Exit now to load existing datastruct.\n'); 
    else 
        didx = input('Enter # to create new datastruct: \n'); 
        datastruct = sprintf('datastruct_%03d', didx); 
        while ismember(datastruct, existingAnalyses) 
            didx = input('Analysis %s already exists... Choose again.\n', datastruct); 
            datastruct = sprintf('datastruct_%03d', didx); 
        end 
        dstructPath = fullfile(analysisDir, datastruct); 
        mkdir(dstructPath); 
        D = struct(); 
    end 
end 
 
D.name = datastruct; 
D.datastructPath = dstructPath; 
D.sourceDir = sourceDir; 
D.acquisitionName = acquisitionName; 
 
D.preprocessing = dsoptions.preprocessing; %preprocessing; 
D.channelIdx = dsoptions.signalchannel; %channelIdx; 
D.extraTiffsExcluded = dsoptions.excludedtiffs; %extraTiffsExcluded; 
D.slices = dsoptions.slices; %slicesToUse; 
D.tefo = dsoptions.tefo; %tefo; 
D.average = dsoptions.averaged; %average; 
 
if D.average 
    D.matchtrials = dsoptions.matchedtiffs; 
end 
 
save(fullfile(dstructPath, datastruct), '-struct', 'D'); 
 
 
fprintf('Created new datastruct analysis: %s\n', D.datastructPath) 

% Add datastruct info to info file:
infotable = struct2table(dsoptions, 'AsArray', true, 'RowNames', {D.name});

% Check previously-made analyses:
analysisinfo_fn = fullfile(analysisDir, 'datastructs.txt');
if exist(analysisinfo, 'file')
    % Load it and check it:
    previousInfo = readtable(analysisinfo);
    allInfo = [previousInfo; infotable];
else
    allInfo = infotable; 

end

writetable(allInfo, analysinfo_fn, 'WriteRowNames', true, 'Delimiter', '\t');
type analysisinfo_fn



