function [RGBimg, masksRGBimg] = createRGBimg(avgimg, maskcell)

nRois = length(maskcell);

RGBimg = zeros([size(avgimg),3]);
RGBimg(:,:,1)=0;
RGBimg(:,:,2)=mat2gray(avgimg); %mat2gray(avgY);
RGBimg(:,:,3)=0;

masksRGBimg = RGBimg;
for roi=1:nRois
    masksRGBimg(:,:,3) = masksRGBimg(:,:,3)+0.7*full(maskcell{roi});
end


end