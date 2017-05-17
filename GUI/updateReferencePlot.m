function [handles, D] = updateReferencePlot(handles, D, newReference, showRois)

avgimg = getCurrentSliceImage(handles, D);      % Get avg image for selected slice and file.
meta = load(D.metaPath);

if newReference==1
    fprintf('Loading data for selected slice...\n');
    
    if strcmp(D.roiType, 'pixels')
        RGBimg = zeros([size(avgimg),3]);
        RGBimg(:,:,1)=0;
        RGBimg(:,:,2)=mat2gray(avgimg); %mat2gray(avgY);
        RGBimg(:,:,3)=0;
        
        D.RGBimg = RGBimg;
        D.masksRGBimg = RGBimg;
        D.avgimg = D.masksRGBimg(:,:,2); %avgimg;
        
        npixels = meta.file(1).si.frameWidth * meta.file(1).si.linesPerFrame;
        
        [maskcell, roiIdxs] = getCurrentSliceMasks(handles,D);        % Get masks for currently selected slice. % NOTE:  Masks created could go with different file.
        setappdata(handles.roigui, 'maskcell', maskcell);
        setappdata(handles.roigui, 'roiIdxs', roiIdxs);
        
        setRoiMax(handles, 1:npixels);                      % Set Max values for ROI entries.
        
        nrois = length(maskcell);
        
    else

        %   avgimg = getCurrentSliceImage(handles, D);
        [maskcell, roiIdxs] = getCurrentSliceMasks(handles,D);        % Get masks for currently selected slice. % NOTE:  Masks created could go with different file.
        setappdata(handles.roigui, 'maskcell', maskcell);
        setappdata(handles.roigui, 'roiIdxs', roiIdxs);

        M = arrayfun(@(roi) reshape(full(maskcell{roi}), [size(maskcell{roi},1)*size(maskcell{roi},2) 1]), 1:length(maskcell), 'UniformOutput', false);
        M = cat(2,M{1:end});
        setappdata(handles.roigui, 'maskmat', M);

        setRoiMax(handles, maskcell);                      % Set Max values for ROI entries.

        [RGBimg, masksRGBimg] = createRGBimg(avgimg, maskcell);         % Create RGB image for average slice and masks.

        D.RGBimg = RGBimg;
        D.masksRGBimg = masksRGBimg;
        D.avgimg = D.masksRGBimg(:,:,2); %avgimg;
        
        nrois = length(maskcell);

    end
    
    fprintf('Done!\n');
end

handles.currSlice.UserData.sliceValue = handles.currSlice.Value;
handles.runMenu.UserData.runValue = handles.runMenu.Value;
handles.currRoi.UserData.currRoi = str2double(handles.currRoi.String);

selectedROI = str2double(handles.currRoi.String);

maskcell = getappdata(handles.roigui,'maskcell');
roiIdxs = getappdata(handles.roigui, 'roiIdxs');
nrois = length(maskcell);


if selectedROI > nrois
    handles.currRoi.Value = nrois;
    handles.currRoi.String = num2str(nrois);
    selectedROI = str2double(handles.currRoi.String);
end

if strcmp(D.roiType, 'pixels')
    handles.roi3D.String = mat2str(maskcell{selectedROI});
else
    handles.roi3D.String = num2str(roiIdxs(selectedROI));
end

%D.masksRGBimg(:,:,1) = D.masksRGBimg(:,:,1).*0;
%D.masksRGBimg(:,:,2) = avgimg; 
%D.masksRGBimg(:,:,3) = D.masksRGBimg(:,:,3).*0;

D.RGBimg(:,:,1) = D.RGBimg(:,:,1).*0;
%D.RGBimg(:,:,2) = avgimg;
D.RGBimg(:,:,3) = D.RGBimg(:,:,3).*0;

cmin = min(D.masksRGBimg(:)); %max(avgimg(:))*0.010;
cmax = max(D.masksRGBimg(:)); %1; %max(avgimg(:));

if selectedROI>0
    if strcmp(D.roiType, 'pixels')
        rx = maskcell{selectedROI}(1);
        ry = maskcell{selectedROI}(2);
        D.masksRGBimg(rx, ry, 1) = D.masksRGBimg(rx, ry, 1) + 0.7;
        D.RGBimg(rx, ry, 1) = D.masksRGBimg(rx, ry, 1) + 0.7;
    else
        D.masksRGBimg(:,:,1) = D.masksRGBimg(:,:,1)+0.7*full(maskcell{selectedROI});
        %D.masksRGBimg(:,:,3) = D.masksRGBimg(:,:,3)+0.7*full(maskcell{selectedROI});
        D.RGBimg(:,:,1) = D.RGBimg(:,:,1)+0.7*full(maskcell{selectedROI});
        %D.RGBimg(:,:,3) = D.RGBimg(:,:,3)+0.7*full(maskcell{selectedROI});
    end
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
            %if newReference==1
                if isfield(D, 'tefo') && D.tefo
                    handles.avgimg = imagesc2(D.masksRGBimg, [cmin, cmax]);
                else
                    handles.avgimg = imagesc2(scalefov(D.masksRGBimg), [cmin, cmax]); %, handles.ax1); %, 'Parent',handles.ax1, 'PickableParts','none', 'HitTest','off');%imagesc(D.masksRGBimg);
                end
                set(gca,'YDir','reverse')
                %set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);

%             else
%                 if isfield(D, 'tefo') && D.tefo
%                     handles.avgimg = imagesc2(D.masksRGBimg);
%                 else
%                     handles.avgimg = imagesc2(scalefov(D.masksRGBimg)); %imshow(D.masksRGBimg);
%                 end
%                 set(gca,'YDir','reverse')
%             end
        else
            %if newReference==1
                if D.tefo
                    handles.avgimg = imagesc2(D.RGBimg, [cmin, cmax]);
                else
                    handles.avgimg = imagesc2(scalefov(D.RGBimg), [cmin, cmax]); %, 'Parent',handles.ax1, 'PickableParts','none', 'HitTest','off');%imagesc(D.masksRGBimg); %imshow(D.masksRGBimg);
                %set(handles.avgimg, 'ButtonDownFcn', @ax1_ButtonDownFcn);
                end

%             else
%                 if D.tefo
%                     handles.avgimg = imagesc2(D.RGBimg); % = imshow(D.RGBimg);
%                 else
%                     handles.avgimg = imagesc2(scalefov(D.RGBimg)); % = imshow(D.RGBimg);
%                 end
%             end
        end
end
%caxis([max(avgimg)*0.10, max(avgimg)]);
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