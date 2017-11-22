function [mean_pixel_corr] = get_pixel_time_correlation(maskcell,Y)
%takes in:
%   maskcell - cell array with masks matrices
%   Y - image stack

%returns
%   mean_pixel_corr - vector with pixel-to-pixel time correlations averaged
%   across pixels for each ROI

    %initialize emptry matrix
    mean_pixel_corr = zeros(1,length(maskcell));
    for r = 1 : length(maskcell)
        %get mask and indices to extract
        mask0 = maskcell{r};
        [pixY,pixX] = find(mask0>0);

        %extract pixel traces
        pix_trace = zeros(length(pixX),size(Y,3));
        for p = 1:length(pixX)
            pix_trace(p,:) = Y(pixY(p), pixX(p), :);
        end

        %get pixel-to-pixel correlation matrix
        R = corr(pix_trace');

        %get mean pixel-to-pixel correlation 
        R_ind = find(triu(ones(size(R)),1)==1);
        mean_pixel_corr(r)=mean(R(R_ind));
    end
   
