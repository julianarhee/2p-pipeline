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

    % Set defaults for ROI values: ----------------------------------------
    handles.currRoi.String = num2str(1); %num2str(round(str2double(handles.currRoi.String)));
    handles.currRoiSlider.Value = str2double('1'); %str2double(handles.currRoi.String);

    % Set options for SLICE menu: -----------------------------------------
    sliceNames = cell(1,length(D.slices));
    for s=1:length(sliceNames)
        sliceNames{s} = sprintf('Slice%02d', D.slices(s));
    end
    handles.currSlice.String = sliceNames;

    % Get default slice idx and value: ------------------------------------
    selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
    selectedSlice = D.slices(selectedSliceIdx);

    % Set file name options for RUN menu: ---------------------------------
    fileNames = cell(1,length(meta.file));
    for fN=1:length(fileNames)
        fileNames{fN} = sprintf('File%02d', fN);
    end
    handles.runMenu.String = fileNames;
    
    % Get default selected file/run: --------------------------------------
    selectedFile = handles.runMenu.Value;

    % Populate MAP menu options: --------------------------------------------
    if isfield(D, 'dfStructName')
        dfStruct = load(fullfile(D.guiPrepend, D.outputDir, D.dfStructName));
        setappdata(handles.roigui, 'df', dfStruct);
        if isempty(dfStruct.slice(selectedSlice).file)
            fprinf('No DF struct found for slice %i.\n', selectedSlice);
            noDF = true;
        else
            noDF = false;
        end
    else
        fprintf('No DF struct found in current acquisition.\n');
        noDF = true;
    end
    
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
    handles.mapMenu.String = mapTypes;
    
    % Populate STIM menu options: -----------------------------------------
    %meta = getappdata(handles.roigui, 'meta');
    handles.stimMenu.String = meta.condTypes;
    handles.stimMenu.UserData.stimType = handles.stimMenu.String;

    
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