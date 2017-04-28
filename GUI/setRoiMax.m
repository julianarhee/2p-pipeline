function setRoiMax(handles, maskcell)

nRois = length(maskcell);
if str2num(handles.currRoi.String) > nRois
    handles.currRoi.String = num2str(length(maskcell));
    handles.currRoiSlider.Value = length(maskcell);
end
handles.currRoiSlider.Value = str2num(handles.currRoi.String);
handles.currRoiSlider.Max = nRois;
handles.currRoiSlider.SliderStep = [1/(nRois-1) (1/nRois)*2];


end