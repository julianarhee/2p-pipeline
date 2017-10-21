function mean_trial_corr = get_trial_correlation(s)

    %get info from structure
    nstim = length(fieldnames(s));
    nrois = s.stim01.nrois;

    %initialize emptry matrix
    mean_trial_corr = zeros(nrois,nstim);

    for stim = 1:nstim
        stimid = sprintf('stim%02d',stim) ;

        for roi = 1:nrois
            %get traces
            trial_traces = s.(stimid).traces(:,:,roi);
            
            %get trial-to-trial correlation matrix
            R = corr(trial_traces');
            
            %get mean trial-to-trial correlation, across trials
            R_ind = find(triu(ones(size(R)),1)==1);
            mean_trial_corr(roi,stim)=mean(R(R_ind));
        end
    end
end
