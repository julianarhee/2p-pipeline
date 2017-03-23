function img = scalefov(img)

img = imresize(img, [size(img,1)*2, size(img,2)]);

end