function D = create_datastruct(dsoptions, overwriteflag)

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
    overwrite = false;
else 
    fprintf('*********************************************************\n'); 
    fprintf('WARNING:  Specified datastruct -- %s -- exists. Overwrite?\n', datastruct); 
    if ~overwriteflag   
	 uinput = input('Press Y/n to overwrite or create new: \n', 's')
    else
	uinput = 'Y'
    end
    if strcmp(uinput, 'Y') || overwriteflag 
        D = struct(); 
        fprintf('New datastruct created: %s.\n', datastruct); 
        fprintf('Not yet saved. Exit now to load existing datastruct.\n'); 
        overwrite = true;
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
        overwrite = false;
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

D.roiType = dsoptions.roitype;
D.seedRois = dsoptions.seedrois;
D.maskType = dsoptions.maskshape;
D.roiSource = dsoptions.maskpath;
D.maskfinder = dsoptions.maskfinder;
 
save(fullfile(dstructPath, datastruct), '-struct', 'D'); 
 
 
fprintf('Created new datastruct analysis: %s\n', D.datastructPath) 

% Add datastruct info to info file:
fields = fieldnames(dsoptions);
for field = 1:length(fields)
    if ~isstr(dsoptions.(fields{field}))
        dsoptions.(fields{field}) = mat2str(double(dsoptions.(fields{field})));
    end
end
infotable = struct2table(dsoptions, 'AsArray', true, 'RowNames', {D.name});
analysisinfo_fn_tmp = fullfile(analysisDir, 'datastructsTMP.txt');
writetable(infotable, analysisinfo_fn_tmp, 'WriteRowNames', true, 'Delimiter', '\t');
infotable = readtable(analysisinfo_fn_tmp, 'Delimiter', '\t', 'ReadRowNames', true, 'TreatAsEmpty', {'Na'});
delete(analysisinfo_fn_tmp)
for field=1:length(fields)
    if ~iscell(infotable{1,fields{field}}) && any(find(isnan(infotable{1,fields{field}})))
        % idxs2fix = find(isnan(infotable{1,fields{field}))
        infotable.(fields{field}) = {'NaN'};
    end
end
 
% Check previously-made analyses:
analysisinfo_fn = fullfile(analysisDir, 'datastructs.txt');
if exist(analysisinfo_fn, 'file')
    % Load it and check it:
    previousInfo = readtable(analysisinfo_fn, 'Delimiter', '\t', 'ReadRowNames', true, 'TreatAsEmpty', {'Na'})
    rownames = previousInfo.Properties.RowNames;
    for field=1:length(fields)
	if ~iscell(previousInfo{1, fields{field}}) && any(find(arrayfun(@(d) isnan(previousInfo{d, fields{field}}), 1:length(rownames))))
	    idxs2fix = find(arrayfun(@(d) isnan(previousInfo{d, fields{field}}), 1:length(rownames)));
            for idx = idxs2fix
	    previousInfo(idx,:).(fields{field}) = ['NaN']; %, length(idxs2fix), 1);
            end
	end
    end

    if overwrite
       
        if any(arrayfun(@(r) strcmp(D.name, rownames{r}), 1:length(rownames)))
            previousInfo({D.name},:) = infotable;
            allInfo = previousInfo;
        else
            allInfo = [previousInfo; infotable];
        end    
    else
        allInfo = [previousInfo; infotable];
    end
else
    allInfo = infotable; 

end

writetable(allInfo, analysisinfo_fn, 'WriteRowNames', true, 'Delimiter', '\t');
type(analysisinfo_fn)



