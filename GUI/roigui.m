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

% Last Modified by GUIDE v2.5 18-Mar-2017 20:55:21

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

updateActivityMap(handles, D);
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
[currDatastruct, currPath, ~] = uigetfile();
D = load(fullfile(currPath, currDatastruct));
D.guiPath = currPath;
setappdata(handles.roigui, 'D', D);

% Get and set meta data:
[fp,fn,fe] = fileparts(D.metaPath);
tmpparts = strsplit(D.guiPath, '/');
mPath = fullfile(tmpparts{1:end-2});
meta = load(fullfile(['/' mPath], 'meta', strcat(fn, fe)));
setappdata(handles.roigui, 'meta', meta);

% Set defaults for ROI values:
handles.currRoi.String = num2str(1); %num2str(round(str2double(handles.currRoi.String)));
handles.currRoiSlider.Value = str2double('1'); %str2double(handles.currRoi.String);

% Set options for SLICE menu:
sliceNames = cell(1,length(D.slices));
for s=1:length(sliceNames)
    sliceNames{s} = sprintf('Slice%02d', D.slices(s));
end
handles.currSlice.String = sliceNames;

% Get default slice idx and value:
selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedSlice = D.slices(selectedSliceIdx);

% Set file name options for RUN menu:
fileNames = cell(1,length(meta.file));
for fN=1:length(fileNames)
    fileNames{fN} = sprintf('File%02d', fN);
end
handles.runMenu.String = fileNames;

% Get default selected file/run:
selectedFile = handles.runMenu.Value;

% Assign MAP menu options:
selectedFile = handles.runMenu.Value;
mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
[tmpP, tmpN,~]=fileparts(D.outputDir);
mapStruct = load(fullfile(D.guiPath, tmpN, mapStructName));
mapTypes = fieldnames(mapStruct.file(selectedFile));
handles.mapMenu.String = mapTypes;

% Load and display average image for currently selected file/slice:
avgimg = getCurrentSliceImage(handles, D);      % Get avg image for selected slice and file.
% masks = getCurrentSliceMasks(handles,D);        % Get masks for currently selected slice. % NOTE:  Masks created could go with different file.
% setRoiMax(handles, masks);                      % Set Max values for ROI entries.
% RGBimg = createRGBmasks(avgimg, masks);         % Create RGB image for average slice and masks.
% updateReferencePlot(handles, RGBimg);           % Update reference mask image.
updateReferencePlot(handles, D);

D.avgimg = avgimg;

% Update activity map:
updateActivityMap(handles, D);

setappdata(handles.roigui, 'D', D);

guidata(hObject,handles);




function currSlice_Callback(hObject, eventdata, handles)
% hObject    handle to currSlice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of currSlice as text
%        str2double(get(hObject,'String')) returns contents of currSlice as a double

D = getappdata(handles.roigui,'D');
sliceNames = cell(1,length(D.slices));
for s=1:length(sliceNames)
    sliceNames{s} = sprintf('Slice%02d', D.slices(s));
end
hObject.String = sliceNames;

updateReferencePlot(handles, D);
updateActivityMap(handles, D)

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

[tmpP, tmpN,~]=fileparts(D.outputDir);
selectedSliceIdx = handles.currSlice.Value; %str2double(handles.currSlice.String);
selectedSlice = D.slices(selectedSliceIdx);
selectedFile = handles.runMenu.Value;


mapStructName = sprintf('maps_Slice%02d.mat', selectedSlice); 
mapStruct = load(fullfile(D.guiPath, tmpN, mapStructName));
mapTypes = fieldnames(mapStruct.file(selectedFile));

hObject.String = mapTypes;

updateActivityMap(handles, D);

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

updateActivityMap(handles, D);

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
