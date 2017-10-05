function mov = read_imgdata(tiffpath)

img_info = imfinfo(tiffpath);
m_img = img_info(1).Width;
n_img = img_info(1).Height;
num_images = length(img_info);
mov = cell(num_images, 1);

for i=1:num_images
    mov{i} = double(imread(tiffpath, 'Index', i, 'Info', img_info));
end

mov = cat(3, mov{1:end});

end
