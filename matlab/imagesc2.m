function h = imagesc2 ( img_data, varargin)
% a wrapper for imagesc, with some formatting going on for nans
% if length(varargin)>0
%     ax = varargin{1};
% else
%     ax = axes();
% end
% plotting data. Removing and scaling axes (this is for image plotting)
%axes(ax)
h = imagesc(img_data);
%h = imagesc(img_data);
axis image off

% setting alpha values
if ndims( img_data ) == 2
  set(h, 'AlphaData', ~isnan(img_data))
elseif ndims( img_data ) == 3
  set(h, 'AlphaData', ~isnan(img_data(:, :, 1)))
end

if nargout < 1
  clear h
end