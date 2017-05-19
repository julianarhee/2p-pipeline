function maskmat = getGolfballs(centers, radii, volumesize, view_sample)

% r=4;
% phi=linspace(0,pi,120);
% theta=linspace(0,2*pi,120);
% [phi,theta]=meshgrid(phi,theta);
% 
% x=r*sin(phi).*cos(theta);
% y=r*sin(phi).*sin(theta);
% z=r*cos(phi); 


d1 = volumesize(1);
d2 = volumesize(2);
d3 = volumesize(3);

[cols rows slices] = meshgrid(1:d1, 1:d2, 1:d3);

% centers = [5,5,5; 15,15,15];
% radii = [2, 3];

spherefunc = @(r) sqrt((rows - round(centers(r,2))).^2 + (cols - round(centers(r,1))).^2 + (slices - round(centers(r,3))).^2) < radii(r).^2;

nrois = length(radii);
masks = arrayfun(@(roi) spherefunc(roi), 1:nrois, 'UniformOutput', false);

% To view:
if view_sample
    maskarray = masks{1}; % roi=1
    patch(isosurface(smooth3(maskarray)), 'FaceColor', 'blue', 'EdgeColor', 'none')
    view(3); 
    axis vis3d %tight
    set(gca,'XLim',[1,d1],'YLim',[1,d2],'ZLim',[1,d3]);
    camlight left; 
    lighting gouraud
end

% Reshape to mat:
maskmat = cellfun(@(roimask) reshape(roimask, d1*d2*d3, []), masks, 'UniformOutput', 0);
maskmat = cat(2, maskmat{1:end});



end

