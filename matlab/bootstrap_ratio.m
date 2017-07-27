
fft1 = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/output/fft3D_File001.mat');
fft2 = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/output/fft3D_File002.mat');
fft3 = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/output/fft3D_File003.mat');
fft4 = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/output/fft3D_File004.mat');

% ratio = 
% 
%                max: [0.0737 0.1059 0.1372 0.1252]
%                min: [6.2633e-04 0.0010 0.0015 7.2878e-04]
%               mean: [0.0123 0.0166 0.0199 0.0187]
%                std: [0.0149 0.0224 0.0284 0.0272]
%     abovethreshold: [185 202 224 209]
%              nrois: [248 248 248 248]
%              
             
threshold = 0.01;
             
nrois = 248;
maxratios = zeros(1,nrois);
for roi=1:nrois
   maxratios(roi) = max([fft1.ratioMatNMF(roi), fft2.ratioMatNMF(roi), fft3.ratioMatNMF(roi), fft4.ratioMatNMF(roi)]); 
end
n_actual = length(find(maxratios>=threshold)) / nrois;

%%

allvals = [fft1.ratioMatNMF, fft2.ratioMatNMF, fft3.ratioMatNMF, fft4.ratioMatNMF];
nvals = length(allvals);
nconds = 4;

niter = 1000;

% Assume normal distN, calculate mag-ratio as p-val:
actual_mean = nanmean(allvals);
actual_std = nanstd(allvals);

r = actual_mean + actual_std.*randn(niter,1);
figure(); hist(abs(r),100);
[H, STATS] = cdfplot(abs(r));


% Bootstrap, random draw and take max. Calculate p that % expressing >=
% actual value:
threshold = 0.01;
percent_active = zeros(1,niter);
for n=1:niter
    
   % draw 4 runs for each cell:
   draw = zeros(1,nrois);
   for roi=1:nrois
      shuffler = randi(nvals, 1, nconds);
      draw(roi) = max(allvals(shuffler));
   end
   
   percent_active(n) = length(find(draw>=threshold)) / nrois;
   
end

figure(); hist(percent_active);

pval = length(find(percent_active >= n_actual)) / niter;
hold on;
title(sprintf('p = %s', num2str(pval)));

%%

% Bootstrap, random draw and take max. Calculate p that % expressing >=
% actual value:
FFT = struct();
FFT.fft1 = fft1;
FFT.fft2 = fft2;
FFT.fft3 = fft3;
FFT.fft4 = fft4;

niter = 10000000;
threshold = 0.01;
percent_active = zeros(1,niter);
[nfreqs, nrois] = size(fft1.magMatNMF);
target_idx = fft1.targetFreqIdx;
for n=1:niter
    
    % Shuffle magnitudes, calculate ratio-mag:
    ratios_by_cond = zeros(nconds, nrois);
    for run=1:nconds
        curr_run = sprintf('fft%i', run);
        magtmp = num2cell(FFT.(curr_run).magMatNMF, 1);
        shuffled_freqs = randi(nfreqs, 1, nfreqs);
        shuffled_mags = cellfun(@(r) r(shuffled_freqs), magtmp, 'UniformOutput', false);
        shuffled_ratios = cellfun(@(r) r(target_idx)/(sum(r)-r(target_idx)), shuffled_mags);
        ratios_by_cond(run, :) = shuffled_ratios;
    end
    draw = max(ratios_by_cond, [], 1);
    
    percent_active(n) = length(find(draw>=threshold)) / nrois;
   
end

figure(); hist(percent_active);

pval = length(find(percent_active >= n_actual)) / niter;
hold on;
title(sprintf('p = %s', num2str(pval)));


%%


% Bootstrap, random draw and take max. Calculate p that % expressing >=
% actual value:
FFT = struct();
FFT.fft1 = fft1;
FFT.fft2 = fft2;
FFT.fft3 = fft3;
FFT.fft4 = fft4;

niter = 10000;
threshold = 0.01;
percent_active = zeros(1,niter);
[nfreqs, nrois] = size(FFT.fft1.magMatNMF);
target_idx = FFT.fft1.targetFreqIdx;

pvals = zeros(nconds, nrois);
for run=1:nconds
    curr_run = sprintf('fft%i', run);
    for roi=1:nrois
        actual_ratio = FFT.(curr_run).ratioMatNMF(roi);
        samples = zeros(1, niter);
        for n=1:niter
            shuffled_freqs = randi(nfreqs, 1, nfreqs);
            shuffled_mags = FFT.(curr_run).magMatNMF(shuffled_freqs, roi);
            samples(n) = shuffled_mags(target_idx) / (sum(shuffled_mags) - shuffled_mags(target_idx));
        end
        pvals(run, roi) = length(find(samples >= actual_ratio)) / niter;
    end
end

pthresh = 0.05
cells = find(min(pvals, [], 1) < pthresh);
ncells = length(cells);
percent_excluded = 1 - (ncells/nrois)

D = load('/nas/volume1/2photon/RESDATA/TEFO/20161219_JR030W/retinotopyFinal/analysis/datastruct_014/datastruct_014.mat');

masks = load(D.maskInfo.maskPaths{1});
cellIDs = masks.roi3Didxs(cells);


for n=1:niter
    
    % Shuffle magnitudes, calculate ratio-mag:
    shuffled_ratios_by_cond = zeros(nconds, nrois);
    for run=1:nconds
        curr_run = sprintf('fft%i', run);
        true_ratios_by_cond(run, :) = FFT.(curr_run).ratioMatNMF;
        magtmp = num2cell(FFT.(curr_run).magMatNMF, 1);
        shuffled_freqs = randi(nfreqs, 1, nfreqs);
        shuffled_mags = cellfun(@(r) r(shuffled_freqs), magtmp, 'UniformOutput', false);
        shuffled_ratios = cellfun(@(r) r(target_idx)/(sum(r)-r(target_idx)), shuffled_mags);
        shuffled_ratios_by_cond(run, :) = shuffled_ratios;
    end
    draw = max(ratios_by_cond, [], 1);
    
    percent_active(n) = length(find(draw>=threshold)) / nrois;
   
end

figure(); hist(percent_active);

pval = length(find(percent_active >= n_actual)) / niter;
hold on;
title(sprintf('p = %s', num2str(pval)));