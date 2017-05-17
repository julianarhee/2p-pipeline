function [RGBimg, masksRGBimg] = createRGBimg(avgimg, maskcell)

nRois = length(maskcell);

RGBimg = zeros([size(avgimg),3]);
RGBimg(:,:,1)=0;
if (max(avgimg(:)) - min(avgimg(:))) > 2000
    newmin = sort(reshape(avgimg, size(avgimg,1)*size(avgimg,2), 1));
    inrange = find(diff(newmin)>5);
    min2use = newmin(inrange(1)+1);
    RGBimg(:,:,2)=mat2gray(avgimg, [min2use, max(avgimg(:))]);
else
    RGBimg(:,:,2)=mat2gray(avgimg); %mat2gray(avgY);
end
RGBimg(:,:,3)=0;

masksRGBimg = RGBimg;
for roi=1:nRois
    masksRGBimg(:,:,3) = masksRGBimg(:,:,3)+0.7*full(maskcell{roi});
end


end