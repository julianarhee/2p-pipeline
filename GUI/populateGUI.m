function [handles, firstLoad] = populateGUI(handles, tmpCurrPath, tmpCurrDatastruct)

firstLoad=0;
loading=1;
while loading==1
    if tmpCurrPath==0
        break;
    end
    tload = tic();
    fprintf('Populating GUI...');
    if (tic()-tload) >= 2
        fprintf('...');
        tload = tic();
    end
    
    % Load main analysis MAT: ---------------------------------------------
    currPath = tmpCurrPath;
    currDatastruct = tmpCurrDatastruct;
    D = load(fullfile(currPath, currDatastruct));
    setappdata(handles.roigui, 'D', D);

    % Get and set PATHS: --------------------------------------------------
    % This matters if using linux or mac, for ex.
    tmpparts = strsplit(currPath, '/');
    if ismac
        D.guiPrepend = fullfile(tmpparts{1:3});
        D.guiPrepend = ['/' D.guiPrepend];
    else
        D.guiPrepend = '';
    end
    setappdata(handles.roigui, 'D', D);
    
    % Load META data and set as appdata: ----------------------------------
    [fp,fn,fe] = fileparts(D.metaPath);
    if strcmp(tmpparts{end}, '')
        mPath = fullfile(tmpparts{1:end-2});
    else
        mPath = fullfile(tmpparts{1:end-1});
    end
    meta = load(fullfile(['/' mPath], 'meta', strcat(fn, fe)));
    
    setappdata(handles.roigui, 'meta', meta);
    fprintf('Meta data loaded.\n');
    
    % Set gui colors:
    guicolors.red = [1 0 0];
    guicolors.lightgray = [0.7 0.7 0.7];
    guicolors.darkgray = [0.3 0.3 0.3];
    setappdata(handles.roigui, 'guicolors', guicolors);
    
    % Set defaults for ROI values: ----------------------------------------
    handles.currRoi.String = num2str(1); %num2str(round(str2double(handles.currRoi.String)));
    handles.currRoiSlider.Value = str2double('1'); %str2double(handles.currRoi.String);

    
    % Set options for SLICE menu: -----------------------------------------
    if ~isfield(D, 'slices')
        sliceNames = cell(1,length(D.slicesToUse));
    else
        sliceNames = cell(1,length(D.slices));
    end
    for s=1:length(sliceNames)
        sliceNames{s} = sprintf('Slice%02d', D.slices(s)); % - D.slices(1) + 1);
    end
    handles.currSlice.String = sliceNames;

    
    % Get default slice idx and value: ------------------------------------
    selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
    if selectedSliceIdx > length(D.slices)
        selectedSliceIdx = length(D.slices);
    end
    selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;

    
    % Set file name options for RUN menu: ---------------------------------
    fileNames = cell(1,length(meta.file));
    for fN=1:length(fileNames)
        fileNames{fN} = sprintf('File%02d', fN);
    end
    handles.runMenu.String = fileNames;
    
    
    % Get default selected file/run: --------------------------------------
    if handles.runMenu.Value > length(handles.runMenu)
        handles.runMenu.Value = length(handles.runMenu);
    end
    selectedFile = handles.runMenu.Value;

    
    % Populate MAP menu options: --------------------------------------------
    if isfield(D, 'dfStructName')
        fprintf('Loading DF datastruct...\n');
        dfStruct = load(fullfile(D.guiPrepend, D.outputDir, D.dfStructName));
        if isempty(dfStruct.slice(selectedSlice).file)
            fprintf('No DF struct found for slice %i.\n', selectedSlice);
            noDF = true;
        else
            noDF = false;
        end
    else
        fprintf('No DF struct found in current acquisition.\n');
        noDF = true;
        dfStruct = struct();
    end
    setappdata(handles.roigui, 'df', dfStruct);

    
    mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
    if ~exist(fullfile(D.guiPrepend, D.outputDir, mapStructName), 'file')
        mapTypes = {'maps - notfound'};
    else
        mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
        mapTypes = fieldnames(mapStruct.file(selectedFile));
    end
    if noDF %isempty(dfStruct.slice(selectedSlice).file)
        mapTypes{end+1} = 'maxDf - not found';
    else
        mapTypes{end+1} = 'maxDf';
    end
    handles.mapMenu.String = sort_nat(mapTypes);

    
    
    % Populate STIM menu options: -----------------------------------------
    %meta = getappdata(handles.roigui, 'meta');
    handles.stimMenu.String = sort_nat(meta.condTypes);
    handles.stimMenu.UserData.stimType = handles.stimMenu.String;
    if handles.stimMenu.Value > length(handles.stimMenu.String)
        handles.stimMenu.Value = 1;
    end
    handles.stimMenu.UserData.currStimName =  handles.stimMenu.String{handles.stimMenu.Value};
    if ~strcmp(D.stimType, 'bar')
%         nStimuli = length(meta.condTypes);
%         colors = zeros(nStimuli,3);
%         for c=1:nStimuli
%             colors(c,:,:) = rand(1,3);
%         end
        stimcolors = meta.stimcolors;
        setappdata(handles.roigui, 'stimcolors', stimcolors);
    
        stimstruct = load(fullfile(D.outputDir, D.stimStructName));
        setappdata(handles.roigui, 'stimstruct', stimstruct);
        trialstruct = load(fullfile(D.outputDir, D.trialStructName));
        setappdata(handles.roigui, 'trialstruct', trialstruct);
        
    else
        fftStructName = sprintf('fft_Slice%02d.mat', selectedSlice); 
        fftStruct = load(fullfile(D.guiPrepend, D.outputDir, fftStructName));
        stimcolors = repmat(guicolors.red, [50,1]);
        setappdata(handles.roigui, 'stimcolors', stimcolors);
        
    end
    
    handles.ax3.UserData.clickedTrial = 1;
    
    
    % Populate Time-Course menu options: ----------------------------------
    if isfield(dfStruct.slice(selectedSlice).file(selectedFile), 'dfMatInferred')
        timecourseTypes = {'dF/F', 'raw', 'processed', 'inferred'};
    else
        timecourseTypes = {'dF/F', 'raw', 'processed'};
    end
    handles.timecourseMenu.String = timecourseTypes;


    % Update UserData for selected path: ----------------------------------
    handles.selectDatastructPush.UserData.currPath = currPath;
    handles.selectDatastructPush.UserData.currStruct = currDatastruct;
    
    % Update text displaying path: ----------------------------------------
    handles.currDatastruct.String = fullfile(currPath, currDatastruct);
    handles.currDatastruct.UserData.currString = handles.currDatastruct.String;
    
    fprintf('Done!\n');
    firstLoad = 1;
    
    loading = 0;
end


end