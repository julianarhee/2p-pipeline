function legends = makeLegends(outputDir)

sim_acquisition = zeros(10,50);
sim_acquisition(:,1) = ones(size(sim_acquisition(:,1)));
legend_im = zeros(10,50,100);
legend_im(:,:,1) = sim_acquisition(:,:,1);
tmpA = sim_acquisition;
for lidx=2:100
    legend_im(:,:,lidx) = circshift(tmpA, 1, 2);
    tmpA = legend_im(:,:,lidx);
end

Fs_lgd = 1;
T_lgd = 1/Fs_lgd;
L_lgd = size(legend_im,3);
t_lgd = (0:L_lgd-1)*T_lgd;
targetFreq_lgd = 1/50;
legend_phase = zeros(size(legend_im,1), size(legend_im,2));
for r_lgd=1:size(legend_im,1)
    for c_lgd=1:size(legend_im,2)
        y_lgd = legend_im(r_lgd,c_lgd,:);
        NFFT_lgd = length(y_lgd);
        legend_ft = fft(y_lgd,NFFT_lgd);
        %freqs = Fs*(0:1/(NFFT/2))/NFFT;
        freqs_lgd = ((0:1/NFFT_lgd:1-1/NFFT_lgd)*Fs_lgd).';
        freq_idx_lgd = find(abs((freqs_lgd-targetFreq_lgd))==min(abs(freqs_lgd-targetFreq_lgd)));
        magY = abs(legend_ft);
        legend_phase(r_lgd,c_lgd) = angle(legend_ft(freq_idx_lgd)); % unwrap(angle(Y(freq_idx)));
    end        
end
% figure()
% imagesc(legend_phase)
% colormap hsv
% caxis([-pi, pi])
% %colorbar()
% axis('off')

legends = struct();
legends.left = legend_phase;
legends.right = fliplr(legend_phase);

% HORIZONTAL:
sim_acquisition = zeros(10,50);
sim_acquisition(1,:) = ones(size(sim_acquisition(1,:)));
legend_im = zeros(10,50,100);
legend_im(:,:,1) = sim_acquisition(:,:,1);
tmpA = sim_acquisition;
for lidx=2:100
    legend_im(:,:,lidx) = circshift(tmpA, 1,1);
    tmpA = legend_im(:,:,lidx);
end

Fs_lgd = 1;
T_lgd = 1/Fs_lgd;
L_lgd = size(legend_im,3);
t_lgd = (0:L_lgd-1)*T_lgd;
targetFreq_lgd = 1/10;
legend_phase = zeros(size(legend_im,1), size(legend_im,2));
for r_lgd=1:size(legend_im,1)
    for c_lgd=1:size(legend_im,2)
        y_lgd = legend_im(r_lgd,c_lgd,:);
        NFFT_lgd = length(y_lgd);
        legend_ft = fft(y_lgd,NFFT_lgd);
        %freqs = Fs*(0:1/(NFFT/2))/NFFT;
        freqs_lgd = ((0:1/NFFT_lgd:1-1/NFFT_lgd)*Fs_lgd).';
        freq_idx_lgd = find(abs((freqs_lgd-targetFreq_lgd))==min(abs(freqs_lgd-targetFreq_lgd)));
        magY = abs(legend_ft);
        legend_phase(r_lgd,c_lgd) = angle(legend_ft(freq_idx_lgd)); % unwrap(angle(Y(freq_idx)));
    end        
end
% figure()
% imagesc(legend_phase)
% colormap hsv
% caxis([-pi, pi])
% %colorbar()
% axis('off')

legends.top = legend_phase;
legends.bottom = flipud(legend_phase);
legend_struct = 'retinotopy_legends';

%FFT.legends = legends;
save(fullfile(outputDir, legend_struct), '-struct', 'legends', '-v7.3');

end