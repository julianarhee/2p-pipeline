function setRoiMax(handles, maskcell)

nRois = length(maskcell);
if handles.currRoi.String > nRois
    handles.currRoi.String = num2str(1);
    handles.currRoiSlider.String = num2str(1);
end

handles.currRoiSlider.Max = nRois;
handles.currRoiSlider.SliderStep = [1/(nRois-1) (1/nRois)*2];


end