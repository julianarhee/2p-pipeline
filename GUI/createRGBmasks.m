function RGBimg = createRGBmasks(avgimg, masks)

nRois = size(masks,3);

RGBimg = zeros([size(avgimg),3]);
RGBimg(:,:,1)=0;
RGBimg(:,:,2)=mat2gray(avgimg); %mat2gray(avgY);
RGBimg(:,:,3)=0;

for roi=1:nRois
    RGBimg(:,:,3) = RGBimg(:,:,3)+0.7*masks(:,:,roi);
end


end