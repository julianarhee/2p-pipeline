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

% Last Modified by GUIDE v2.5 19-Mar-2017 19:58:45

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
[handles, D] = updateReferencePlot(handles, D, newReference, showRois)

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

updateActivityMap(handles, D, meta);
% [tmpP, tmpN,~]=fileparts(D.outputDir);
% selectedSlice = str2double(handles.currSlice.String);
% selectedFile = handles.runMenu.Value;
% 
% mapStructName = sprintf('maps_Slice%02d', selectedSlice); 
% mapStruct = load(fullfile(D.guiPath, tmpN, mapStructName));
% selectedMapType = handles.
% displayMap = mapStruct.file(selectedFile).(selectedMapType);

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
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);

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
setappdata(handles.roigui, 'D', D);

% Update activity map:
updateActivityMap(handles, D, meta);

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

if ~isfield(hObject.UserData, 'sliceValue')
    newReference=1;
elseif hObject.UserData.sliceValue ~= hObject.Value
    newReference=1;
else
    newReference=0;
end
[handles, D] = updateReferencePlot(handles, D, newReference, showRois);
set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

updateActivityMap(handles, D, meta)

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
selectedSlice = D.slices(selectedSliceIdx);
selectedFile = handles.runMenu.Value;
%currRunName = meta.file(selectedFile).mw.runName;

mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
mapStruct = load(fullfile(D.guiPrepend, D.outputDir, mapStructName));
mapTypes = fieldnames(mapStruct.file(selectedFile));

hObject.String = mapTypes;

updateActivityMap(handles, D, meta);

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

updateActivityMap(handles, D, meta);

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

if ~strcmp(hObject.String, fullfile(sPath, sStruct))
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
    showRois = handles.roiToggle.Value;
    [handles, D] = updateReferencePlot(handles, D, newReference, showRois);
    setappdata(handles.roigui, 'D', D);

    % Update activity map:
    updateActivityMap(handles, D, meta);

    % Set UserData fields:
    handles.currSlice.UserData.sliceValue = handles.currSlice.Value;
    handles.runMenu.UserData.runValue = handles.runMenu.Value;

    % Update appdata for D struct:
    setappdata(handles.roigui, 'D', D);

    guidata(hObject,handles);
end

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

%handles.avgimg.HitTest = 'off'

S.ax = hObject.Parent;
S.axp = S.ax.Position;
S.xlm = S.ax.XLim;
S.ylm = S.ax.YLim;
S.dfx = diff(S.xlm);
S.dfy = diff(S.ylm);

F = S.ax.CurrentPoint  % The current point w.r.t the figure.

Cx =  S.xlm(1) + (F(1)-S.axp(1)).*(S.dfx/S.axp(3));
Cy =  S.xlm(1) + (F(2)-S.axp(2)).*(S.dfy/S.axp(4));
%fprintf('Curr position: %s\n', num2str([Cx,Cy],2))

% Figure out of the current point is over the axes or not -> logicals.
tf1 = S.axp(1) <= F(1) && F(1) <= S.axp(1) + S.axp(3);
tf2 = S.axp(2) <= F(2) && F(2) <= S.axp(2) + S.axp(4);

if tf1 && tf2
    % Calculate the current point w.r.t. the axes.
    Cx =  S.xlm(1) + (F(1)-S.axp(1)).*(S.dfx/S.axp(3));
    Cy =  S.xlm(1) + (F(2)-S.axp(2)).*(S.dfy/S.axp(4));
    %set(S.tx(2),'str',num2str([Cx,Cy],2))
    fprintf('Curr position: %s\n', num2str([Cx,Cy],2))
end
