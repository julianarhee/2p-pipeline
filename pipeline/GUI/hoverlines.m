% Convert current 
% Test As:
% 
%  plot(1:50, peaks(50))
%  hoverlines(gca);
%
function hoverlines( varargin )
% takes one or zero arguments
switch nargin
    case 0, ax = gca;
    case 1, ax = varargin{1};
    otherwise, error('bad args');
end
% get figure
fig = get(ax,'parent');
% get all line children of the axes
lines = findobj( get(ax,'children'), 'type', 'line');
if isempty(lines), 
    warning('no lines, nothing done');
    return
end
% set hover behavior
setbehavior(lines);
% reset all sitting
for h = lines(:)'
    d = get(h,'userdata');
    d.sit();
end
set(ax,'userdata',lines(1));
set(fig,'windowButtonMotionFcn',@hover)
end

%%
function hover(fig, ignore)
% click
rob = java.awt.Robot();
rob.mousePress(java.awt.event.InputEvent.BUTTON1_MASK);
rob.mouseRelease(java.awt.event.InputEvent.BUTTON1_MASK);

% get object under cursor
hovering = get(fig,'currentObject');
% return if hovering over same object
lastone = get(gca,'userdata');
if hovering == lastone, return; end
% get behavior data 
hData = get(hovering,'userdata');
% hovering over some other type of object perhaps
if ~isfield( hData, 'sit' ), return; end
% ok, stand up
hData.stand();
% sit-down previous
hData = get(lastone,'userdata');
hData.sit();
% store as lastone
set(gca,'userData',hovering);
end

%% Set hover behavior
function setbehavior( hs )
    for h = hs(:)'
        high = get(h,'color');
        dim  = rgb2hsv(high);
        dim(2) = 0.1;
        dim(3) = 0.9;
        dim = hsv2rgb(dim);
        
        hov.sit   = @() set(h ...
            ,'color', dim ...
            ,'linewidth', 1 ...
            ... ,'marker', 'none' ...
            );
        hov.stand = @() set(h ...
            ,'color', high ...
            ,'linewidth', 2 ...
            ... ,'marker', '*' ...
            );
        set(h,'userdata',hov);
    end
end

