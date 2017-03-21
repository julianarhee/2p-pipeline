function setRoiMax(handles, masks)

nRois = size(masks,3);
handles.currRoi.String = num2str(1);
handles.currRoiSlider.String = num2str(1);

handles.currRoiSlider.Max = nRois;
handles.currRoiSlider.SliderStep = [1/(nRois-1) (1/nRois)*2];


end