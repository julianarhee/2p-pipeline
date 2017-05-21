function varargout = roigui(varargin)
% ROIGUI MATLAB code for roigui.fig
%      ROIGUI, by itself, creates a new ROIGUI or raises the existing
%      singleton*.
%
%      H = ROIGUI returns the handle to a new ROIGUI or the handle to
%      the existing singleton*.
%
%      ROIGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ROIGUI.M with the given input arguments.
%
%      ROIGUI('Property','Value',...) creates a new ROIGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before roigui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to roigui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help roigui

% Last Modified by GUIDE v2.5 09-May-2017 16:32:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @roigui_OpeningFcn, ...
                   'gui_OutputFcn',  @roigui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before roigui is made visible.
function roigui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to roigui (see VARARGIN)

% Choose default command line output for roigui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes roigui wait for user response (see UIRESUME)
% uiwait(handles.roigui);


% --- Outputs from this function are returned to the command line.
function varargout = roigui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function currRoi_Callback(hObject, eventdata, handles)
% hObject    handle to currRoi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of currRoi as text
%        str2double(get(hObject,'String')) returns contents of currRoi as a double

hObject.String = num2str(round(str2double(hObject.String)));
handles.currRoiSlider.Value = str2double(hObject.String);

newReference = 0;
showRois = handles.roiToggle.Value;
D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui,'meta');
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end

roiIdxs = getappdata(handles.roigui, 'roiIdxs');
handles.roi3D.String = num2str(roiIdxs(str2double(hObject.String)));

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function currRoi_CreateFcn(hObject, eventdata, handles)
% hObject    handle to currRoi (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in stimMenu.
function stimMenu_Callback(hObject, eventdata, handles)
% hObject    handle to stimMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns stimMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from stimMenu
D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui, 'meta');
hObject.String = sort_nat(meta.condTypes);
hObject.UserData.stimType = hObject.String;
hObject.UserData.currStimValue = hObject.Value;
hObject.UserData.currStimName =  hObject.String{hObject.Value};


% TODO: Function to update stimulus-trace plot (PSTH):
[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function stimMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to stimMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in runMenu.
function runMenu_Callback(hObject, eventdata, handles)
% hObject    handle to runMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns runMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from runMenu

D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui, 'meta');

fileNames = cell(1,length(meta.file));
for fN=1:length(fileNames)
    fileNames{fN} = sprintf('File%02d', fN);
end
hObject.String = fileNames;
hObject.UserData.runValue = hObject.Value;

switch D.roiType
    case '3Dcnmf'
        newReference=1;
    case 'manual3Drois'
        newReference=1;     
    otherwise
        if ~isfield(handles.currSlice.UserData, 'sliceValue')
            newReference=1;
        elseif handles.currSlice.UserData.sliceValue ~= handles.currSlice.Value
            newReference=1;
        else
            newReference=0;
        end
end
showRois = handles.roiToggle.Value;
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

[handles, D] = updateActivityMap(handles, D, meta);
set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
    if handles.stimShowAvg.Value
        set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
    else
        set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
    end
    handles.ax3.Children = flipud(handles.ax3.Children);

end

% [tmpP, tmpN,~]=fileparts(D.outputDir);
% selectedSlice = str2double(handles.currSlice.String);
% selectedFile = handles.runMenu.Value;
% 
% mapStructName = sprintf('maps_Slice%02d', selectedSlice); 
% mapStruct = load(fullfile(D.guiPath, tmpN, mapStructName));
% selectedMapType = handles.
% displayMap = mapStruct.file(selectedFile).(selectedMapType);
% Update appdata for D struct:
setappdata(handles.roigui, 'D', D);


guidata(hObject,handles);



% --- Executes during object creation, after setting all properties.
function runMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to runMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function currRoiSlider_Callback(hObject, eventdata, handles)
% hObject    handle to currRoiSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
hObject.Value = round(hObject.Value);
handles.currRoi.String = num2str(hObject.Value);

newReference = 0;
showRois = handles.roiToggle.Value;
D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui,'meta');

[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end


roiIdxs = getappdata(handles.roigui, 'roiIdxs');
handles.roi3D.String = num2str(roiIdxs(hObject.Value));

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function currRoiSlider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to currRoiSlider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in selectDatastructPush.
function selectDatastructPush_Callback(hObject, eventdata, handles)
% hObject    handle to selectDatastructPush (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[tmpCurrDatastruct, tmpCurrPath, ~] = uigetfile();

% If legitimate path/struct chosen, populate GUI with components. This
% will:
% - Create appdata:  D, meta
% - Update handles and specific UserData fields.
[handles, firstLoad] = populateGUI(handles, tmpCurrPath, tmpCurrDatastruct);

% Load and display average image for currently selected file/slice:
D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui, 'meta');

if firstLoad==1
    newReference=1;
elseif ~isfield(hObject.UserData, 'currPath')
    newReference=1;
elseif ~strcmp(hObject.UserData.currPath, fullfile(D.guiPrepend, D.datastructPath))
    newReference=1;
else
    newReference=0;
end
showRois = handles.roiToggle.Value;
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

setappdata(handles.roigui, 'D', D);

% Update activity map:
[handles, D] = updateActivityMap(handles, D, meta);
set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

% Update Timecourse plot:
[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

% Update Stimulus plot:
[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end

% Set UserData fields:
handles.currSlice.UserData.sliceValue = handles.currSlice.Value;
handles.runMenu.UserData.runValue = handles.runMenu.Value;

% Update appdata for D struct:
setappdata(handles.roigui, 'D', D);


guidata(hObject,handles);




function currSlice_Callback(hObject, eventdata, handles)
% hObject    handle to currSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of currSlice as text
%        str2double(get(hObject,'String')) returns contents of currSlice as a double

D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui, 'meta');

%selectedFile = handles.runMenu.Value;
%currRunName = meta.file(selectedFile).mw.runName;

sliceNames = cell(1,length(D.slices));
for s=1:length(sliceNames)
    sliceNames{s} = sprintf('Slice%02d', D.slices(s));
end
hObject.String = sliceNames;

showRois = handles.roiToggle.Value;

% if ~isfield(hObject.UserData, 'sliceValue')
%     newReference=1;
% elseif hObject.UserData.sliceValue ~= hObject.Value
%     newReference=1;
% else
%     newReference=0;
% end
newReference=1;
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

[handles, D] = updateActivityMap(handles, D, meta);
set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end

setappdata(handles.roigui, 'D', D);
hObject.UserData.sliceValue = hObject.Value;

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function currSlice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to currSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in mapMenu.
function mapMenu_Callback(hObject, eventdata, handles)
% hObject    handle to mapMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns mapMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from mapMenu

D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui, 'meta');

selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
if selectedSliceIdx > length(D.slices)
    handles.currSlice.Value = length(D.slices);
    selectedSliceIdx = handles.currSlice.Value;
end
selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
selectedFile = handles.runMenu.Value;
%currRunName = meta.file(selectedFile).mw.runName;

% mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
% mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
% mapTypes = fieldnames(mapStruct.file(selectedFile));

% Populate MAP menu options: --------------------------------------------
if isfield(D, 'dfStructName')
    dfStruct = getappdata(handles.roigui, 'df'); %load(fullfile(D.guiPrepend, D.outputDir, D.dfStructName));
    %setappdata(handles.roigui, 'df', dfStruct);
    if isempty(dfStruct.slice(selectedSliceIdx).file)
        fprintf('No DF struct found for slice %i.\n', selectedSlice);
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
    if noDF
        mapTypes = {'NaN'};
    else
        mapTypes = {'maps - notfound'};
    end
else
    mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
    mapTypes = fieldnames(mapStruct.file(selectedFile));
end
if noDF %isempty(dfStruct.slice(selectedSlice).file)
    mapTypes{end+1} = 'maxDf - not found';
else
    mapTypes{end+1} = 'maxDf';
end
%handles.mapMenu.String = mapTypes;

hObject.String = mapTypes;

[handles, D] = updateActivityMap(handles, D, meta);
set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);


guidata(hObject,handles);



% --- Executes during object creation, after setting all properties.
function mapMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mapMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function threshold_Callback(hObject, eventdata, handles)
% hObject    handle to threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of threshold as text
%        str2double(get(hObject,'String')) returns contents of threshold as a double

hObject.String = num2str(str2double(hObject.String));

D = getappdata(handles.roigui,'D');
meta = getappdata(handles.roigui, 'meta');

[handles, D] = updateActivityMap(handles, D, meta);
set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function threshold_CreateFcn(hObject, eventdata, handles)
% hObject    handle to threshold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in roiToggle.
function roiToggle_Callback(hObject, eventdata, handles)
% hObject    handle to roiToggle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of roiToggle

D = getappdata(handles.roigui,'D');

showRois = hObject.Value;
if handles.currSlice.UserData.sliceValue ~= handles.currSlice.Value
    newReference=1;
else
    newReference=0;
end
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

setappdata(handles.roigui, 'D', D);

guidata(hObject,handles);



function currDatastruct_Callback(hObject, eventdata, handles)
% hObject    handle to currDatastruct (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of currDatastruct as text
%        str2double(get(hObject,'String')) returns contents of currDatastruct as a double

sPath = handles.selectDatastructPush.UserData.currPath;
sStruct = handles.selectDatastructPush.UserData.currStruct;

[txtpath,txtstruct,txtext] = fileparts(hObject.String);

%if ~strcmp(hObject.String, fullfile(sPath, sStruct))
    % do stuff
    [handles, firstLoad] = populateGUI(handles, txtpath, strcat(txtstruct, txtext));

    % Load and display average image for currently selected file/slice:
    D = getappdata(handles.roigui,'D');
    meta = getappdata(handles.roigui, 'meta');
    
    if firstLoad==1
        newReference=1;
    elseif ~isfield(hObject.UserData, 'currPath')
        newReference=1;
    elseif ~strcmp(hObject.UserData.currPath, fullfile(D.guiPrepend, D.datastructPath))
        newReference=1;
    else
        newReference=0;
    end
    
    if newReference==1
        cla(handles.ax1);
        cla(handles.ax2);
    end
    showRois = handles.roiToggle.Value;
    [handles, D] = updateReferencePlot(handles, D, newReference, showRois);
    set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);
    setappdata(handles.roigui, 'D', D);

    % Update activity map:
    [handles, D] = updateActivityMap(handles, D, meta);
    set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

    % Update timecourse plot:
    [handles, D] = updateTimeCourse(handles, D, meta);
    nEpochs = handles.ax4.UserData.trialEpochs;
    set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);
    
    % Update stimulus plot:
    [handles, D] = updateStimulusPlot(handles, D);
    if ~strcmp(D.stimType, 'bar')
    if handles.stimShowAvg.Value
        set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
    else
        set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
    end
    end
    handles.ax3.Children = flipud(handles.ax3.Children);


    % Set UserData fields:
    handles.currSlice.UserData.sliceValue = handles.currSlice.Value;
    handles.runMenu.UserData.runValue = handles.runMenu.Value;

    % Update appdata for D struct:
    setappdata(handles.roigui, 'D', D);

    guidata(hObject,handles);
%end

hObject.String = fullfile(handles.selectDatastructPush.UserData.currPath,...
                            handles.selectDatastructPush.UserData.currStruct);

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function currDatastruct_CreateFcn(hObject, eventdata, handles)
% hObject    handle to currDatastruct (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on mouse press over axes background.
function ax1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to ax1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% WindowButtonMotionFcn for the figure.
handles = guidata(hObject);
%handles.avgimg.HitTest = 'off'

%hitpoint = eventdata.IntersectionPoint;

currPoint = eventdata.IntersectionPoint;

D = getappdata(handles.roigui, 'D');
if ~isfield(D, 'tefo') || ~D.tefo
    if strcmp(D.maskType, 'cnmf')
        if D.maskInfo.params.scaleFOV
            cp = [round(currPoint(1)) round(currPoint(2))];
        else
            cp = [round(currPoint(1)) round(currPoint(2)/2)]; 
        end
    else
        cp = [round(currPoint(1)) round(currPoint(2)/2)]; %get(hsandles.ax1,'CurrentPoint')
    end
else
    cp = [round(currPoint(1)) round(currPoint(2))];
end

if strcmp(D.roiType, 'pixels')
    % TODO:  FIX this so that size of image is not hard-coded, and fix
    % secondary ROI field ('roi3d') to read x,y coordinates:
    
    meta = load(D.metaPath);
    d1 = meta.file(1).si.frameWidth;
    d2 = meta.file(1).si.linesPerFrame;

    roiMatch = sub2ind([d1, d2], cp(2), cp(1)); % image is flipped
    
else
    maskcell = getappdata(handles.roigui,'maskcell');
    % M = arrayfun(@(roi) reshape(full(maskcell{roi}), [size(maskcell{roi},1)*size(maskcell{roi},2) 1]), 1:length(maskcell), 'UniformOutput', false);
    % M = cat(2,M{1:end});
    M = getappdata(handles.roigui, 'maskmat');

    colsubscript = sub2ind(size(maskcell{1}), cp(2), cp(1));
    [i,j,s] = find(M);
    roiIdx = find(i==colsubscript);
    if length(roiIdx)>1
        roiIdx = roiIdx(1);
        roiMatch = j(roiIdx);
    elseif length(roiIdx)==0
        roiMatch = handles.currRoi.UserData.currRoi; 
        fprintf('Try again, you did not select an ROI...\n');
    else
        roiMatch = j(roiIdx);
    end
end

%fprintf('ROI: %i\n', roiMatch);
handles.currRoi.String = num2str(roiMatch);
handles.currRoi.Value = roiMatch;
handles.currRoiSlider.Value = str2double(handles.currRoi.String);

newReference = 1;
showRois = handles.roiToggle.Value;
%D = getappdata(handles.roigui,'D');
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

%TODO
% ********************
% Update "stimulus PSTH" plot
meta = getappdata(handles.roigui, 'meta');

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end


guidata(hObject,handles);


% --- Executes on button press in stimShowAvg.
function stimShowAvg_Callback(hObject, eventdata, handles)
% hObject    handle to stimShowAvg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of stimShowAvg
D = getappdata(handles.roigui, 'D');
meta = getappdata(handles.roigui, 'meta');

[handles, D] = updateStimulusPlot(handles, D);
if hObject.Value
    handles.stimMenu.Enable = 'off';
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    handles.stimMenu.Enable = 'on';
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);


if isfield(handles.ax3.UserData, 'clickedStim')
    handles.stimMenu.Value = handles.ax3.UserData.clickedStim;
end

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

% [handles, D] = updateStimulusPlot(handles, D);
% 
% if strcmp(handles.stimMenu.Enable, 'on')
%     set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
% end


guidata(hObject,handles)




% --- Executes on button press in stimShowAll.
function stimShowAll_Callback(hObject, eventdata, handles)
% hObject    handle to stimShowAll (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

D = getappdata(handles.roigui, 'D');
selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
selectedRoi = str2double(handles.currRoi.String);

stimstruct = getappdata(handles.roigui, 'stimstruct');
stimcolors = getappdata(handles.roigui, 'stimcolors');
guicolors = getappdata(handles.roigui, 'guicolors');

nStim = length(fieldnames(stimstruct.slice(selectedSlice)));
if nStim > 20
    nRows = 5;
else
    nRows = 2;
end

% if ~isfield(handles, 'allstimPSTH')
%     fh = figure();
% else
%     fh = figure(fh);
% end
fh=figure();

% if ~ishandle(handles.allstimPSTH)
%     fh=figure();
% else
%     fh = figure(handles.allstimPSTH);
% end

dflim = str2double(handles.dfrange.String);

%you can include anything you want in here as long as you specify the parent as
%being fh
pidx = 1;
for stimidx=1:nStim
    
    currstim = sprintf('stim%i', stimidx);
    mwTrialTimes = stimstruct.slice(selectedSlice).(currstim).mwTrialTimes;
    dfMat = stimstruct.slice(selectedSlice).(currstim).dfTraceCell{selectedRoi}.*100;
    if handles.smooth.Value
        for tridx=1:size(dfMat, 2)
            dfMat(:,tridx) = smooth(dfMat(:,tridx), 'rlowess');
        end
    end
        
    %dfMat = stimstruct.slice(selectedSlice).(currstim).dfCell{selectedRoi}.*100;
    mwSec = stimstruct.slice(selectedSlice).(currstim).siTimeMat; % If created before 03/29/2017, need to transpose

    subaxis(nRows, ceil(nStim/nRows), pidx, 'Spacing', 0.02, 'Padding', 0.01, 'Margin', 0.05)
    cla;
    plot(mwSec, dfMat, 'Color', guicolors.lightgray, 'LineWidth',0.1);
    hold on;

    % Plot MW stim ON patch:
    stimOnset = mean(mwTrialTimes(:,2));
    stimOffset = mean(mwTrialTimes(:,3));
    
    xlim([-1.2, 3.2]);
    ylim([-dflim, dflim]);
    
    ylims = get(gca, 'ylim');
    
    v = [stimOnset ylims(1); stimOffset ylims(1);...
        stimOffset ylims(2); stimOnset ylims(2)];
    f = [1 2 3 4];
    patch('Faces',f,'Vertices',v,'FaceColor',stimcolors(stimidx,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold on;

    % Plot MEAN trace for current ROI across stimulus reps:
    %meanTraceInterp = mean(dfMat, 1);
    meanTraceInterp = nanmean(dfMat, 2);
    meanMWTime = nanmean(mwSec,2);
    plot(meanMWTime, meanTraceInterp, 'k', 'LineWidth', 2);
    hold on;
    
    if pidx==(ceil(nStim/nRows)*nRows - ceil(nStim/nRows) + 1) || pidx==1
        % stuff
        if pidx==1
            set(gca, 'xticklabel', []);
            set(gca, 'xtick', []);
        else
            set(gca, 'TickDir', 'out');
            xlabel('time (s)')
            ylabel('df/f');
        end
    else
        %axis 'off'
        set(gca, 'xticklabel', []);
        set(gca, 'xtick', []);
        set(gca, 'yticklabel', []);   
        set(gca, 'ytick', []);
        set(gca, 'xcolor', 'w');
        set(gca, 'ycolor', 'w');
        
    end
    set(gca, 'Box', 'off');
    
    title(currstim, 'FontSize', 8);
    
    pidx = pidx + 1;
    
end

suptitle(sprintf('ROI %i', selectedRoi));
handles.allstimPSTH = fh;
    
guidata(hObject, handles)

% --- Executes on selection change in timecourseMenu.
function timecourseMenu_Callback(hObject, eventdata, handles)
% hObject    handle to timecourseMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns timecourseMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from timecourseMenu
D = getappdata(handles.roigui, 'D');
meta = getappdata(handles.roigui, 'meta');

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] =  updateActivityMap(handles, D, meta);
set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);

guidata(hObject,handles)


% --- Executes during object creation, after setting all properties.
function timecourseMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to timecourseMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% 
% % --- Executes on mouse press over axes background.
% function stimplotTrial_ButtonDownFcn(hObject, eventdata, handles)
% % hObject    handle to ax1 (see GCBO)
% % eventdata  reserved - to be defined in a future version of MATLAB
% % handles    structure with handles and user data (see GUIDATA)
% 
% % WindowButtonMotionFcn for the figure.
% handles = guidata(hObject);
% clickedTrial = handles.stimplot.UserData.clickedTrial;
% fprintf('Selected trial: %i', clickedTrial);
% 
% 
% 
% guidata(hObject, handles)


% --- Executes on mouse press over axes background.
function ax3_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to ax3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = guidata(hObject);
dfmat = getappdata(handles.roigui, 'dfMat');

sitWidth=0.5;
standWidth=2;


nTrials = size(dfmat,2);

if handles.stimShowAvg.Value
    lines = findobj(handles.stimtrialmean, 'type', 'line');
    for stimAvgTraceIdx=1:length(dfmat)
        currAvgTrace = dfmat{stimAvgTraceIdx};
        if length(currAvgTrace)~=length(hObject.YData)
            continue;
        else
            if sum(hObject.YData==currAvgTrace.') > length(hObject.YData)*.7
                clickedTrial = stimAvgTraceIdx;
            end
        end
    end
    fprintf('Selected stimulus: %i\n', clickedTrial);
    title(sprintf('Stim %i selected\n', clickedTrial),'FontSize', 8);     
    handles.ax3.UserData.clickedStim = clickedTrial; % bad naming...
    handles.stimMenu.Value = handles.ax3.UserData.clickedStim;
    
else
    lines = findobj(handles.stimtrials, 'type', 'line');
    nTrials = size(dfmat,2);
    
    if strcmp(hObject.Type, 'line')
        % Transpose dfmat to get each trial as a row, then just take the
        % first found value (since this will be equal to the actual trial,
        % i.e., first frame/tpoint of selected trial):
        clickedTrial = find(ismember(dfmat.', hObject.YData));
        clickedTrial = clickedTrial(1); 
        fprintf('Selected trial: %i\n', clickedTrial);
        title(sprintf('Trial %i of %i selected', clickedTrial, nTrials), 'FontSize', 8);
    end

end

handles.ax3.Children = flipud(handles.ax3.Children);

if eventdata.Button==3    
    % Turn off previously standing line if different one selected:
    if isfield(handles.ax3.UserData, 'clickedTrial') && clickedTrial~=handles.ax3.UserData.clickedTrial
        lines(handles.ax3.UserData.clickedTrial).LineWidth=sitWidth;
        lines(clickedTrial).LineWidth=standWidth;
    elseif isfield(handles.ax3.UserData, 'clickedTrial') && clickedTrial==handles.ax3.UserData.clickedTrial && lines(clickedTrial).LineWidth==standWidth
        % TUrn off previously standing line if current one re-clicked:
        lines(clickedTrial).LineWidth=sitWidth;
    else
        lines(clickedTrial).LineWidth=standWidth;
        
    end
   handles.ax3.UserData.viewTrial = false; 
   
else
    % Highlight the currently selected (LEFT-click) trial of current
    % stimulus and update other plots to view the file/run in which trial
    % occurred:
    %if ~handles.stimShowAvg.Value
    handles.ax3.UserData.viewTrial = true;
    %else
    %    handles.ax3.UserData.viewTrial = false;
    %end
    lines(clickedTrial).LineWidth=standWidth;
    if isfield(handles.ax3.UserData, 'clickedTrial') && clickedTrial~=handles.ax3.UserData.clickedTrial
        lines(handles.ax3.UserData.clickedTrial).LineWidth=sitWidth;
    end
    if isfield(handles.ax4.UserData, 'stimPlotIdx') && clickedTrial~=handles.ax4.UserData.stimPlotIdx
        lines(handles.ax4.UserData.stimPlotIdx).LineWidth=sitWidth;
    end
end

% Update currently selected trial:
handles.ax3.UserData.clickedTrial = clickedTrial;

% Update time-course and activity maps if specific trial is selected to
% match the TIFF (i.e., file/run) in which the selected trial occurred:
if handles.ax3.UserData.viewTrial

    trialstruct = getappdata(handles.roigui, 'trialstruct');
    meta = getappdata(handles.roigui, 'meta');
    D = getappdata(handles.roigui, 'D');

    selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
    selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
    selectedStimIdx = handles.stimMenu.Value;
    stimNames = handles.stimMenu.String;
    selectedStim = handles.stimMenu.String{selectedStimIdx};
    selectedRoi = str2double(handles.currRoi.String);
    
    if handles.stimShowAvg.Value
        clickedTrialRun = trialstruct.slice(selectedSlice).info.(selectedStim){1}.tiffNum;
        clickedTrialIdxInRun = trialstruct.slice(selectedSlice).info.(selectedStim){1}.trialIdxInRun;
    else
        clickedTrialRun = trialstruct.slice(selectedSlice).info.(selectedStim){clickedTrial}.tiffNum;
        clickedTrialIdxInRun = trialstruct.slice(selectedSlice).info.(selectedStim){clickedTrial}.trialIdxInRun;
    end
    handles.runMenu.Value = clickedTrialRun;
    handles.ax3.UserData.clickedTrialIdxInRun = clickedTrialIdxInRun;
    handles.ax3.UserData.clickedTrialRun = clickedTrialRun;
    
    [handles, D] = updateTimeCourse(handles, D, meta);
    nEpochs = handles.ax4.UserData.trialEpochs;
    set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);
    
    [handles, D] =  updateActivityMap(handles, D, meta);
    set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

end

uistack(lines, 'top');

guidata(hObject, handles);

    
    


% --- Executes on mouse motion over figure - except title and menu.
function roigui_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to roigui (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% handles = guidata(hObject);
% % Try this for highlting ROIs with hover only over ax1:
% relPanelPos = handles.panelAx3.Position;
% S.ax = handles.ax3;
% S.axp = S.ax.Position;
% 
% S.axp(1) = S.axp(1)+relPanelPos(1);
% S.axp(2) = S.axp(2)+relPanelPos(2);
% %S.axp(2) = S.axp(1) + S.axp(2);
% %S.axp(3) = S.axp(3)+relPanelPos(3);
% %S.axp(4) = S.axp(3) + S.axp(3);
% 
% 
% S.xlm = S.ax.XLim;
% S.ylm = S.ax.YLim;
% 
% S.dfx = diff(S.xlm);
% S.dfy = diff(S.ylm);
% 
% F = hObject.CurrentPoint;  % The current point w.r.t the figure.
% 
% % Figure out of the current point is over the axes or not -> logicals.
% tf1 = S.axp(1) <= F(1) && F(1) <= S.axp(1) + S.axp(3);
% tf2 = S.axp(2) <= F(2) && F(2) <= S.axp(2) + S.axp(4);
% 
% if tf1 && tf2
%     % Calculate the current point w.r.t. the axes.
%     Cx =  S.xlm(1) + (F(1)-S.axp(1)).*(S.dfx/S.axp(3));
%     Cy =  S.ylm(1) + (F(2)-S.axp(2)).*(S.dfy/S.axp(4));
%     %set(S.tx(2),'str',num2str([Cx,Cy],2))
%     %fprintf('Curr position: %s\n', num2str([Cx,Cy],2));
%     %hover(handles.ax3);
%     
%     % get all line children of the axes
% %     lines = findobj(handles.stimtrials, 'type', 'line');
% %     if isempty(lines), 
% %         warning('no lines, nothing done');
% %         return
% %     else
% %     
% %     % set hover behavior
% %     setbehavior(lines);
% %     
% %     % reset all sitting
% %     if ~isempty(handles.ax3.UserData)
% %         if handles.ax3.UserData.viewTrial
% %         % do nothing
% %         else
% %         tstart = tic();
% %         while toc(tstart) < 1
% %             % do nothing
% %         end
% %         for h = lines(:)'
% %             d = get(h,'userdata');
% %             d.sit();
% %         end
% %         end
% %     end
% %     
% %         
% %     end
%     
% %     if isfield(handles.ax3, 'UserData')
% %         if ~isempty(handles.ax3.UserData.clickedTrial)
% %         d = get(lines(clickedTrial), 'userdata');
% %         d.stand();
% %         end
% %     end
% %     prevbestmatch = 1;
% %     bestmatch = [];
% %     
% %     candidatesX = zeros(1,length(lines));
% %     candidatesY = zeros(1,length(lines));
% % 
% %     lidx=1;
% %     for line=lines(:)'
% %         abdiffsY = abs(line.YData-Cy);
% %         abdiffsX = abs(line.XData-Cx);
% %         candidateY = find(abs(line.YData-Cy)==min(abs(line.YData-Cy)));
% %         candidateX = find(abs(line.XData-Cx)==min(abs(line.XData-Cx)));
% %         if min(abdiffsY)+min(abdiffsX) < 1
% %             candidatesX(lidx) = abdiffsX(candidateX);
% %             candidatesY(lidx) = abdiffsY(candidateY);
% %         end
% %         lidx = lidx+1;
% %     end
% %     bestmatch = [bestmatch find(candidatesY==min(candidatesY))];
% %     
% %         
% %     %end
% %     
% %     for hix = 1:length(lines(:)');
% %         h = lines(hix);
% %         d = get(h,'userdata');
% %         if hix==bestmatch(1) %hix==bestmatch(1)
% %             d.stand();
% %         else
% %         	d.sit();
% %         end
% %         
% %     end
% %    
% 
% %     set(handles.stimtrials,'userdata',lines(1));
% %     
% %     ttimer = tic();
% %     while toc(ttimer) < 1
% %         lastone = get(handles.stimtrials, 'userdata');
% %         lidx=1;
% %         candidates = zeros(length(lines), 2);
% %         for line=lines(:)'
% %             abdiffs = abs(line.YData-Cy);
% %             candidate = find(abs(line.YData-Cy)==min(abs(line.YData-Cy)));
% %             candidates(lidx,:) = [lidx, abdiffs(candidate)];
% %             lidx = lidx+1;
% %         end
% %         bestmatch = find(min(candidates(:,2)));
% %         hovering = lines(bestmatch); %get(handles.stimtrials(bestmatch),'userdata');
% %         
% %         if hovering == lastone, return; end
% %         % get behavior data 
% %         hData = get(hovering,'userdata');
% %         % hovering over some other type of object perhaps
% %         if ~isfield( hData, 'sit' ), return; end
% %         % ok, stand up
% %         hData.stand();
% %         % sit-down previous
% %         hData = get(lastone,'userdata');
% %         hData.sit();
% %         % store as lastone
% %         set(handles.stimtrials,'userdata',hovering);
% %     end
% 
%     
% 
% %     end
% 
% end
% %     
%     
% function setbehavior( hs )
%     for h = hs(:)'
% %         high = get(h,'color');
% %         dim  = rgb2hsv(high);
% %         dim(2) = 0.1;
% %         dim(3) = 0.9;
% %         dim = hsv2rgb(dim);
%         
%         hov.sit   = @() set(h,'linewidth', .5);
%         
%         hov.stand = @() set(h,'linewidth', 2);
%         set(h,'userdata',hov);
%     end


% --- Executes on mouse press over axes background.
function ax4_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to ax4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = guidata(hObject);
D = getappdata(handles.roigui, 'D');
meta = getappdata(handles.roigui, 'meta');
trialstruct = getappdata(handles.roigui, 'trialstruct');

sitWidth=0.5;
standWidth=2;


oldStimVal = handles.stimMenu.Value;

selectedSliceIdx = handles.currSlice.Value;
selectedSlice = D.slices(selectedSliceIdx); % - D.slices(1) + 1;
selectedFile = handles.runMenu.Value;
currRunName = meta.file(selectedFile).mw.runName;

mwTimes = meta.file(selectedFile).mw.mwSec;
mwTimes = mwTimes(1:2:end);
nTrials = length(mwTimes);
mwCodes = meta.file(selectedFile).mw.pymat.(currRunName).stimIDs;
mwCodes = mwCodes(1:2:end);

colors = getappdata(handles.roigui, 'stimcolors');

guicolors = getappdata(handles.roigui, 'guicolors');

patches = findobj(handles.mwepochs, 'type', 'patch');

if strcmp(hObject.Type, 'patch')
    clickedTrial = find(ismember(mwTimes, hObject.Vertices(1,1)));
    clickedTrial = clickedTrial(1);
    %fprintf('Selected trial: %i\n', clickedTrial);
    title(sprintf('Trial %i of %i selected', clickedTrial, nTrials), 'FontSize', 8);
end

clickedTrialStim = mwCodes(clickedTrial);
% Check if there's more than 1 trial:
stimReps = find(mwCodes==clickedTrialStim);
clickedTrialStimColor = colors(clickedTrialStim,:);
handles.stimMenu.Value = clickedTrialStim;
      
[handles, D] = updateStimulusPlot(handles, D);
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);

% Highlight selected stim, if exists:
standAlpha = 0.8;
sitAlpha =0.4;
if handles.stimShowAvg.Value
    if isfield(handles.ax4.UserData, 'clickedTrialStim') ...
            && clickedTrialStim~=handles.ax4.UserData.clickedTrialStim
        handles.stimtrialmean(clickedTrialStim).LineWidth = standWidth;
        handles.stimtrialmean(handles.ax4.UserData.clickedTrialStim).LineWidth = sitWidth;
        uistack(handles.stimtrialmean(clickedTrialStim), 'top');
        
        patches(clickedTrial).FaceAlpha = standAlpha;
        patches(handles.ax4.UserData.clickedTrial).FaceAlpha = sitAlpha;
        
        
    elseif isfield(handles.ax4.UserData, 'clickedTrialStim') ...
            && clickedTrial==handles.ax4.UserData.clickedTrial ...
            && patches(clickedTrial).FaceAlpha==standAlpha
            
        % If same TRIAL picked, and corresponding mean-stim trace still on,
        % turn off previously standing line if re-clicked:
        handles.stimtrialmean(clickedTrialStim).LineWidth = sitWidth;
        patches(clickedTrial).FaceAlpha = sitAlpha;
        
    else
        handles.stimtrialmean(clickedTrialStim).LineWidth = standWidth;
        uistack(handles.stimtrialmean(clickedTrialStim), 'top');
        
        patches(clickedTrial).FaceAlpha = standAlpha;

    end
    
    stimPlotIdx = clickedTrial;

else
    % Highlight corresponding trial on stim-plot:
    tmptrials = trialstruct.slice(selectedSlice).info.(['stim' num2str(clickedTrialStim)]);
    tiffidxs = cellfun(@(tmptrial) tmptrial.tiffNum, tmptrials);
    stimTrialReps = cellfun(@(foundtrial) foundtrial.currStimRep, tmptrials(tiffidxs==selectedFile));
    stimPlotIdx = stimTrialReps(clickedTrial==stimReps);

    if isfield(handles.ax4.UserData, 'clickedTrialStim') ...
            && clickedTrial~=handles.ax4.UserData.clickedTrial
        % Turn off old selected patch and trials, if new one selected:
        handles.stimtrials(stimPlotIdx).LineWidth = standWidth;
        %handles.stimtrials(handles.ax4.UserData.stimPlotIdx).LineWidth = sitWidth;
        title(sprintf('Trial %i of %i selected', stimPlotIdx, length(handles.stimtrials)), 'FontSize', 8);
        
        patches(clickedTrial).FaceColor = clickedTrialStimColor;
        patches(clickedTrial).FaceAlpha = standAlpha;
        isOn = true;
        
        if clickedTrialStim~=handles.ax4.UserData.clickedTrialStim
            patches(handles.ax4.UserData.clickedTrial).FaceColor = guicolors.lightgray;
        end
        patches(handles.ax4.UserData.clickedTrial).FaceAlpha = sitAlpha;
        
    elseif isfield(handles.ax4.UserData, 'clickedTrialStim') ...
            && clickedTrial==handles.ax4.UserData.clickedTrial ...
            && patches(handles.ax4.UserData.clickedTrial).FaceAlpha==standAlpha
        % Turn off previously standing line if current one re-clicked:
        handles.stimtrials(handles.ax4.UserData.stimPlotIdx).LineWidth = sitWidth;
        patches(handles.ax4.UserData.clickedTrial).FaceAlpha = sitAlpha;
        isOn = false;
    else
        handles.stimtrials(stimPlotIdx).LineWidth = standWidth;
        title(sprintf('Trial %i of %i selected', stimPlotIdx, length(handles.stimtrials)), 'FontSize', 8);
        
        patches(clickedTrial).FaceColor = clickedTrialStimColor;
        patches(clickedTrial).FaceAlpha = standAlpha;
        isOn = true;
    end

    % Update selected trial face patch to match stim-color:
%     patches(clickedTrial).FaceColor = clickedTrialStimColor;
%     patches(clickedTrial).FaceAlpha = standAlpha;
%     if isfield(handles.ax4.UserData, 'clickedTrialStim') && clickedTrialStim ~= handles.ax4.UserData.clickedTrialStim
%         patches(handles.ax4.UserData.clickedTrial).FaceColor = guicolors.lightgray;
%         patches(handles.ax4.UserData.clickedTrial).FaceAlpha = sitAlpha;
%     end
    if clickedTrialStim ~= oldStimVal
        prevStimTrialIdxs = find(mwCodes==oldStimVal);
        for ptrial = prevStimTrialIdxs
            patches(ptrial).FaceColor = guicolors.lightgray;
            patches(ptrial).FaceAlpha = sitAlpha;
        end
    end
    
    uistack(handles.stimtrials, 'top');
end
                

handles.ax4.UserData.clickedTrialStim = clickedTrialStim;
handles.ax4.UserData.clickedTrial = clickedTrial;
handles.ax4.UserData.stimPlotIdx = stimPlotIdx;

guidata(hObject, handles);


% --- Executes on mouse press over axes background.
function ax2_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to ax2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% WindowButtonMotionFcn for the figure.
handles = guidata(hObject);
%handles.avgimg.HitTest = 'off'

%hitpoint = eventdata.IntersectionPoint;

currPoint = eventdata.IntersectionPoint;

D = getappdata(handles.roigui, 'D');
if ~isfield(D, 'tefo') || ~D.tefo
    if strcmp(D.maskType, 'cnmf')
        if D.maskInfo.params.scaleFOV
            cp = [round(currPoint(1)) round(currPoint(2))];
        else
            cp = [round(currPoint(1)) round(currPoint(2)/2)]; 
        end
    else
        cp = [round(currPoint(1)) round(currPoint(2)/2)]; %get(hsandles.ax1,'CurrentPoint')
    end
else
    cp = [round(currPoint(1)) round(currPoint(2))];
end


if strcmp(D.roiType, 'pixels')
    % TODO:  FIX this so that size of image is not hard-coded, and fix
    % secondary ROI field ('roi3d') to read x,y coordinates:
    meta = load(D.metaPath);
    d1 = meta.file(1).si.frameWidth;
    d2 = meta.file(1).si.linesPerFrame;

    roiMatch = sub2ind([d1, d2], cp(2), cp(1)); % image is flipped
    
else
    maskcell = getappdata(handles.roigui,'maskcell');
    % M = arrayfun(@(roi) reshape(full(maskcell{roi}), [size(maskcell{roi},1)*size(maskcell{roi},2) 1]), 1:length(maskcell), 'UniformOutput', false);
    % M = cat(2,M{1:end});
    M = getappdata(handles.roigui, 'maskmat');

    colsubscript = sub2ind(size(maskcell{1}), cp(2), cp(1));
    [i,j,s] = find(M);
    roiIdx = find(i==colsubscript);
    if length(roiIdx)>1
        roiIdx = roiIdx(1);
        roiMatch = j(roiIdx);
    elseif length(roiIdx)==0
        roiMatch = handles.currRoi.UserData.currRoi; 
        fprintf('Try again, you did not select an ROI...\n');
    else
        roiMatch = j(roiIdx);
    end
    
end

%fprintf('ROI: %i\n', roiMatch);
handles.currRoi.String = num2str(roiMatch);
handles.currRoi.Value = roiMatch;
handles.currRoiSlider.Value = str2double(handles.currRoi.String);

newReference = 1;
showRois = handles.roiToggle.Value;
%D = getappdata(handles.roigui,'D');
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

%TODO
% ********************
% Update "stimulus PSTH" plot
meta = getappdata(handles.roigui, 'meta');

[handles, D] = updateTimeCourse(handles, D, meta);
nEpochs = handles.ax4.UserData.trialEpochs;
set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end


guidata(hObject,handles);



function roi3D_Callback(hObject, eventdata, handles)
% hObject    handle to roi3D (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of roi3D as text
%        str2double(get(hObject,'String')) returns contents of roi3D as a double
% hObject.String = num2str(round(str2double(hObject.String)));
% curr3Droi = str2double(hObject.String);

selectedFile = handles.runMenu.Value;
selectedSliceIdx = handles.currSlice.Value;
selectedSlice = handles.currSlice.String{selectedSliceIdx};

D = getappdata(handles.roigui, 'D');
meta = getappdata(handles.roigui, 'meta');

if strcmp(D.roiType, 'pixels')

    curr3Droi = str2mat(hObject.String);
    newReference=0;
    showRois = handles.roiToggle.Value;
    
    [handles, D] = updateReferencePlot(handles, D, newReference, showRois);
    set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

    im
    [handles, D] = updateActivityMap(handles, D, meta);
    set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);

    %TODO
    % ********************
    % Update "stimulus PSTH" plot
    meta = getappdata(handles.roigui, 'meta');

    [handles, D] = updateTimeCourse(handles, D, meta);
    nEpochs = handles.ax4.UserData.trialEpochs;
    set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

    [handles, D] = updateStimulusPlot(handles, D);
    if ~strcmp(D.stimType, 'bar')
    if handles.stimShowAvg.Value
        set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
    else
        set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
    end
    handles.ax3.Children = flipud(handles.ax3.Children);
    end

    
else
    hObject.String = num2str(round(str2double(hObject.String)));
    curr3Droi = str2double(hObject.String);

    maskcell = getappdata(handles.roigui, 'maskcell');
    roiIdxs = getappdata(handles.roigui, 'roiIdxs');

    if ~ismember(curr3Droi, roiIdxs)
        maskstruct3D = load(D.maskInfo.maskPaths{selectedFile});
        if isfield(maskstruct3D, 'roiIDs')
            % Sometimes IDs are NOT continuous (like when we use EM ids to
            % get ROIs...)
            matidx = find(maskstruct3D.roiIDs == curr3Droi);
            correspondingSlice = maskstruct3D.centers(matidx, 3);
        elseif isfield(maskstruct3D, 'roi3Didxs')
            matidx = find(maskstruct3D.roi3Didxs == curr3Droi);
            correspondingSlice = maskstruct3D.centers(matidx, 3);
            
        else
            correspondingSlice = maskstruct3D.centers(curr3Droi, 3);
        end
        if correspondingSlice ~= selectedSliceIdx
            handles.currSlice.Value = correspondingSlice;
            newReference = 1;
            showRois = handles.roiToggle.Value;
            
            [maskcell, roiIdxs] = getCurrentSliceMasks(handles,D);
            roiSliceID = find(curr3Droi==roiIdxs)
            handles.currRoi.Value = roiSliceID;
            handles.currRoi.String = num2str(roiSliceID);
            handles.currRoiSlider.Value = roiSliceID;
            
            [handles, D] = updateReferencePlot(handles, D, newReference, showRois);
            set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);


            [handles, D] = updateActivityMap(handles, D, meta);
            set(handles.map, 'ButtonDownFcn', @ax2_ButtonDownFcn);
            
            %TODO
            % ********************
            % Update "stimulus PSTH" plot
            meta = getappdata(handles.roigui, 'meta');

            [handles, D] = updateTimeCourse(handles, D, meta);
            nEpochs = handles.ax4.UserData.trialEpochs;
            set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

            [handles, D] = updateStimulusPlot(handles, D);
            if ~strcmp(D.stimType, 'bar')
            if handles.stimShowAvg.Value
                set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
            else
                set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
            end
            handles.ax3.Children = flipud(handles.ax3.Children);
            end
        end
    else
        handles.currRoi.String = num2str(find(roiIdxs==curr3Droi));
        newReference=0;
        showRois = handles.roiToggle.Value;
        [handles, D] = updateReferencePlot(handles, D, newReference, showRois);
        set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

        meta = getappdata(handles.roigui, 'meta');

        [handles, D] = updateTimeCourse(handles, D, meta);
        nEpochs = handles.ax4.UserData.trialEpochs;
        set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);

        [handles, D] = updateStimulusPlot(handles, D);
        if ~strcmp(D.stimType, 'bar')
        if handles.stimShowAvg.Value
            set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
        else
            set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
        end
        handles.ax3.Children = flipud(handles.ax3.Children);
        end

    end
end    
% newReference = 0;
% showRois = handles.roiToggle.Value;
% D = getappdata(handles.roigui,'D');
% meta = getappdata(handles.roigui,'meta');
% [handles, D] = updateReferencePlot(handles, D, newReference, showRois);
% set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);
% 
% [handles, D] = updateTimeCourse(handles, D, meta);
% nEpochs = handles.ax4.UserData.trialEpochs;
% set(handles.mwepochs(1:2:nEpochs), 'ButtonDownFcn', @ax4_ButtonDownFcn);
% 
% [handles, D] = updateStimulusPlot(handles, D);
% if ~strcmp(D.stimType, 'bar')
% if handles.stimShowAvg.Value
%     set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
% else
%     set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
% end
% handles.ax3.Children = flipud(handles.ax3.Children);
% end

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function roi3D_CreateFcn(hObject, eventdata, handles)
% hObject    handle to roi3D (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function dfrange_Callback(hObject, eventdata, handles)
% hObject    handle to dfrange (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of dfrange as text
%        str2double(get(hObject,'String')) returns contents of dfrange as a double

hObject.Value = str2double(hObject.String);


guidata(hObject,handles);


% --- Executes during object creation, after setting all properties.
function dfrange_CreateFcn(hObject, eventdata, handles)
% hObject    handle to dfrange (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in smooth.
function smooth_Callback(hObject, eventdata, handles)
% hObject    handle to smooth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of smooth

D = getappdata(handles.roigui,'D');

[handles, D] = updateStimulusPlot(handles, D);
if ~strcmp(D.stimType, 'bar')
if handles.stimShowAvg.Value
    set(handles.stimtrialmean, 'ButtonDownFcn', @ax3_ButtonDownFcn);
else
    set(handles.stimtrials, 'ButtonDownFcn', @ax3_ButtonDownFcn);
end
handles.ax3.Children = flipud(handles.ax3.Children);
end

guidata(hObject,handles);
