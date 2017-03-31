function [handles, D] = updateReferencePlot(handles, D, newReference, showRois)

avgimg = getCurrentSliceImage(handles, D);      % Get avg image for selected slice and file.

if newReference==1
    fprintf('Loading data for selected slice...\n');
    
    %   avgimg = getCurrentSliceImage(handles, D);
    maskcell = getCurrentSliceMasks(handles,D);        % Get masks for currently selected slice. % NOTE:  Masks created could go with different file.
    setappdata(handles.roigui, 'maskcell', maskcell);
    
    M = arrayfun(@(roi) reshape(full(maskcell{roi}), [size(maskcell{roi},1)*size(maskcell{roi},2) 1]), 1:length(maskcell), 'UniformOutput', false);
    M = cat(2,M{1:end});
    setappdata(handles.roigui, 'maskmat', M);

    setRoiMax(handles, maskcell);                      % Set Max values for ROI entries.

    [RGBimg, masksRGBimg] = createRGBimg(avgimg, maskcell);         % Create RGB image for average slice and masks.
    
    D.RGBimg = RGBimg;
    D.masksRGBimg = masksRGBimg;
    D.avgimg = avgimg;
    
    fprintf('Done!\n');
end

handles.currSlice.UserData.sliceValue = handles.currSlice.Value;
handles.runMenu.UserData.runValue = handles.runMenu.Value;
handles.currRoi.UserData.currRoi = str2double(handles.currRoi.String);

selectedROI = str2double(handles.currRoi.String);
maskcell = getappdata(handles.roigui,'maskcell');
D.masksRGBimg(:,:,1) = D.masksRGBimg(:,:,1).*0;
D.RGBimg(:,:,1) = D.RGBimg(:,:,1).*0;
D.RGBimg(:,:,3) = D.RGBimg(:,:,3).*0;
    
if selectedROI>0
    D.masksRGBimg(:,:,1) = D.masksRGBimg(:,:,1)+0.7*full(maskcell{selectedROI});
    D.RGBimg(:,:,1) = D.RGBimg(:,:,1)+0.7*full(maskcell{selectedROI});
    D.RGBimg(:,:,3) = D.RGBimg(:,:,3)+0.7*full(maskcell{selectedROI});
end

axes(handles.ax1);
%current_cb = get(handles.ax1, 'ButtonDownFcn');
hold on;
% if showRois==1
%     handles.avgimg = imagesc2(D.masksRGBimg);
% else
%     handles.avgimg = imagesc2(D.RGBimg);
% end

switch D.maskType
    case 'nmf'
        % TODO:  Show mask overlay, or use contours?
        % ***************************
        handles.avgimg = plot_contours(maskcell,scalefov(D.masksRGBimg),D.maskInfo.nmfoptions,1);
    otherwise
        if showRois
            if newReference==1
                handles.avgimg = imagesc2(scalefov(D.masksRGBimg)); %, handles.ax1); %, 'Parent',handles.ax1, 'PickableParts','none', 'HitTest','off');%imagesc(D.masksRGBimg);
                set(gca,'YDir','reverse')
                %set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

            else
                handles.avgimg.CData = scalefov(D.masksRGBimg); %imshow(D.masksRGBimg);
                set(gca,'YDir','reverse')
            end
        else
            if newReference==1
                handles.avgimg = imagesc2(scalefov(D.RGBimg)); %, 'Parent',handles.ax1, 'PickableParts','none', 'HitTest','off');%imagesc(D.masksRGBimg); %imshow(D.masksRGBimg);
                %set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

            else
                handles.avgimg.CData = scalefov(D.RGBimg); % = imshow(D.RGBimg);
            end
        end
end

% child_handles = allchild(handles.avgimg);
% set(child_handles,'HitTest','off');


%handles.avgimg.HitTest = 'off'; %'on';
%handles.avgimg.PickableParts = 'none'; %'visible';
%set(handles.ax1, 'ButtonDownFcn',{@ax1_ButtonDownFcn});
%set(handles.avgimg, 'ButtonDownFcn',{@ax1_ButtonDownFcn});


%set(handles.avgimg, 'ButtonDownFcn', @(src,eventdata)ax1_ButtonDownFcn(src,eventdata,handles))

%set(handles.ax1, 'ButtonDownFcn',@(src,eventdata)ax1_ButtonDownFcn(src,eventdata,handles));
%set(handles.avgimg, 'ButtonDownFcn', {@axes1_ButtonDownFcn})
%handles.avgimg.Parent.HitTest = 'off';

% handles.avgimg.Parent.Units = 'pixel';
% 
% S.ax = handles.avgimg.Parent; %handles.avgimg.Parent;
% S.axp = S.ax.Position;
% S.xlm = S.ax.XLim;
% S.ylm = S.ax.YLim;
% S.dfx = diff(S.xlm);
% S.dfy = diff(S.ylm);

%set(S.ax,'ax1_ButtonDownFcn',{@fh_wbmfcn,S}) % Set the motion detector.

end