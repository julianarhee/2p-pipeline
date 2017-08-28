function h = imagesc2 ( img_data, varargin )
% a wrapper for imagesc, with some formatting going on for nans

% plotting data. Removing and scaling axes (this is for image plotting)
if length(varargin)>0
    clim = varargin{1};
    h = imagesc(img_data,clim);
else
    h = imagesc(img_data);
end
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