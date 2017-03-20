function updateReferencePlot(handles, D)

avgimg = getCurrentSliceImage(handles, D);
masks = getCurrentSliceMasks(handles,D);        % Get masks for currently selected slice. % NOTE:  Masks created could go with different file.
setRoiMax(handles, masks);                      % Set Max values for ROI entries.
RGBimg = createRGBmasks(avgimg, masks);         % Create RGB image for average slice and masks.
%updateReferencePlot(handles, RGBimg);           % Update reference mask image.

axes(handles.ax1);
handles.avgimg = imshow(RGBimg);


end