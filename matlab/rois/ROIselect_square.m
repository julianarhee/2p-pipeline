function [masks, RGBimg]=ROIselect_square(calcimg)

%key codes: x-delete chosen centres
%          mouse click-draw ROI
%          Scroll up/down - increase/decrease radius of ROI
%          uparrow-increase brightness of green channel
%          downarrow-decrease brightness of green channel
%          rightarrow-increase brightness of red channel
%          leftarrow-decrease brightness of red channel
%          q-quit the program
close all
quitsig=1;
g=1;    % color value bound for green channel
r=1;    % color value bound for red channel
RGBimg = zeros([size(calcimg),3]);
RGBimg(:,:,1)=0;
RGBimg(:,:,2)=calcimg;
RGBimg(:,:,3)=0;
radius=10;  %in pixels
max_radius = max(size(calcimg))/4;

figure('KeyPressFcn',@KeyDownListener,'WindowButtonMotionFcn',@Wbmf,'WindowButtonDownFcn',@Wbdf,'WindowScrollWheelFcn',@Wswf);
set(gca, 'DataAspectRatio',[1 1 1]); %jyr
set(gca,'XTick',[]); % hides the ticks on x-axis jyr
set(gca,'YTick',[]); % hides the ticks on x-axis
set(gca,'color','None'); % hides the white bckgrnd
set(gca,'XColor','None'); % changes the color of y-axis to white
set(gca,'YColor','None'); % changes the color of y-axis to white

ax=axes;
im_h=imshow(RGBimg,'Parent',ax);
%set(gca,'Units','pixels');
set(gca, 'units', 'normalized', 'position', [0.05 0.15 0.9 0.8]) % Do this to let figure resizing follow
set(gca, 'xlimmode','manual',...
    'ylimmode','manual',...
    'zlimmode','manual')
drawnow;
masks=[];
masks_ones={};
pos=[10,10,2*radius,2*radius];
h = imrect(ax, pos);

mode = -1;

RGBimg(RGBimg>1)=1;
while quitsig
    set(im_h,'CData',RGBimg)
    title(ax,['#cells selected ' num2str(size(masks,3))])
    drawnow;
end
close all

    function KeyDownListener(~,event)
        if strcmp(event.Key,'uparrow')  %to increase the brightness of the green channel
            g=g-0.05;
            if g<0.05
                g=0.05;
            end
            RGBimg(:,:,2)=imadjust(calcimg,[0 g]);
            set(im_h,'CData',RGBimg)
            drawnow;
        elseif strcmp(event.Key,'downarrow')   %to decrease the brightness of the green channel
            g=g+0.05;
            if g>1
                g=1;
            end
            RGBimg(:,:,2)=imadjust(calcimg,[0 g]);
            set(im_h,'CData',RGBimg)
            drawnow;
        
        elseif event.Character=='x'     %to delete detected centres in the ROI selected
            if isempty(masks)
                return
            else
                del_mask=h.createMask;
                del_ind=find(del_mask==1);  %indices of the delete mask
                flag=[];
                k=1;
                for j=1:size(masks,3)
                    
                    if ~isempty(intersect(del_ind,masks_ones{j}))
                        flag(k)=j; %#ok<AGROW>
                        k=k+1;
                        RGBimg(:,:,3)=RGBimg(:,:,3)-0.5*masks(:,:,j);
                        RGBimg(:,:,1)=RGBimg(:,:,1)-0.5*masks(:,:,j);
                    end
                end
                masks(:,:,flag)=[];
                masks_ones(flag)=[];
                RGBimg(RGBimg<0)=0;
                set(im_h,'CData',RGBimg)
                drawnow;
                
            end
        elseif event.Character=='m'%change mode so that user can change ellipse aspect ratio by dragging edge handles
            if mode == 1%create masks before switching out of mod
                if isempty(masks)
                    masks(:,:,1)=h.createMask;
                    masks_ones{1}=find(squeeze(masks(:,:,1))==1);
                    RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,1);
                    RGBimg(:,:,1)=RGBimg(:,:,1)+0.5*masks(:,:,1);
                    set(im_h,'CData',RGBimg)
                    drawnow;
                else
                    numcells=size(masks,3);
                    masks(:,:,numcells+1)=h.createMask;
                    masks_ones{numcells+1}=find(squeeze(masks(:,:,numcells+1))==1);
                    RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,numcells+1);
                    RGBimg(:,:,1)=RGBimg(:,:,1)+0.5*masks(:,:,numcells+1);
                    set(im_h,'CData',RGBimg)
                    drawnow;
                end
            end
            mode=mode*-1;
        elseif event.Character=='q'
            quitsig=0;
        end
    end
    function Wbmf(~,~)
        if mode == -1
            cp = ax.CurrentPoint;
            pos=[cp(1,1)-radius,cp(1,2)-radius,2*radius,2*radius];
            setPosition(h,pos);
        else
            return
        end
    end
    function Wbdf(~,~)  % mouse down function creates masks
        if mode == -1
            if isempty(masks)
                masks(:,:,1)=h.createMask;
                masks_ones{1}=find(squeeze(masks(:,:,1))==1);
                RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,1);
                RGBimg(:,:,1)=RGBimg(:,:,1)+0.5*masks(:,:,1);
                set(im_h,'CData',RGBimg)
                drawnow;
            else
                numcells=size(masks,3);
                masks(:,:,numcells+1)=h.createMask;
                masks_ones{numcells+1}=find(squeeze(masks(:,:,numcells+1))==1);
                RGBimg(:,:,3)=RGBimg(:,:,3)+0.5*masks(:,:,numcells+1);
                RGBimg(:,:,1)=RGBimg(:,:,1)+0.5*masks(:,:,numcells+1);
                set(im_h,'CData',RGBimg)
                drawnow;
            end
        else
            return
        end
    end
    function Wswf(~,callbackdata)   %scrolling up and down function
        if callbackdata.VerticalScrollCount < 0
            if radius<max_radius
                radius=radius+1;
            else
                radius=max_radius;
            end
        elseif callbackdata.VerticalScrollCount > 0
            if radius>1
                radius=radius-1;
            else
                radius=1;
            end
        end
        title(ax,num2str(callbackdata.VerticalScrollCount));
        drawnow
    end

end


