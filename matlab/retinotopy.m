% RETINOTOPY.

% -------------------------------------------------------------------------
% Acquisition Information:
% -------------------------------------------------------------------------
% Get info for a given acquisition for each slice from specified analysis.
% This gives FFT analysis script access to all neeed vars stored from
% create_acquisition_structs.m pipeline.

acquisition_info;

% -------------------------------------------------------------------------
% FFT analysis:
% -------------------------------------------------------------------------
% For each analysed slice, do FFT analysis on each file (i.e., run, or
% condition rep).
% For a given set of traces (on a given run) extracted from a slice, use 
% corresponding metaInfo for maps.

for sidx = slices_to_use
    T = load(fullfile(tracesDir, tracesPaths{sidx}));
    T = T.T;
    clear T.T
    
    for fidx=1:ntiffs
        
        meta = load(metaPaths{fidx});
        
        traces = T.traces.file{fidx};
        masks = T.masks.file{fidx};
        avgY = T.avg_image.file{fidx};

        Fs = meta.si_volume_rate;
        target_freq = meta.target_freq;
        ncycles = meta.ncycles;
        ntotal_slices = meta.nframes_per_volume;
        
        cut_end=1;
        crop = meta.n_true_frames; %round((1/target_freq)*ncycles*Fs);
        winsz = round((1/target_freq)*Fs*2);
        
        switch roi_type
            case 'create_rois'
                [d1,d2,~] = size(T.masks.file{fidx});
                [nrois, tpoints] = size(T.traces.file{fidx});
            case 'pixels'
                %[d1,d2,tpoints] = size(T.traces.file{fidx});
                [d1, d2] = size(avgY);
                tpoints = size(T.traces.file{fidx},3);
        end
        
        % Get phase and magnitude maps:
        phase_map = zeros(d1, d2, 1);
        mag_map = zeros(d1, d2, 1);
        max_map = zeros(d1, d2, 1);

        fft_struct = struct();
        for roi=1:size(traces,1)
            
            slice_indices = slice:meta.nframes_per_volume:meta.ntotal_frames;
            curr_trace = traces(roi, :);
            
            % Subtract rolling mean to get rid of drift:
            % -------------------------------------------------------------
            tmp0 = zeros(1,length(slice_indices));
            if check_slice
                tmp0(:) = squeeze(vol_trace(1:end));
            end
            if cut_end==1
                tmp0 = tmp0(1:crop);
            end
            tmp1 = padarray(tmp0,[0 winsz],tmp0(1),'pre');
            tmp1 = padarray(tmp1,[0 winsz],tmp1(end),'post');
            rollingAvg=conv(tmp1,fspecial('average',[1 winsz]),'same');%average
            rollingAvg=rollingAvg(winsz+1:end-winsz);
            trace_y = tmp0 - rollingAvg;
            
            % Do FFT:
            % -------------------------------------------------------------
            NFFT = length(trace_y);
            fft_y = fft(trace_y,NFFT);
            %F = ((0:1/NFFT:1-1/NFFT)*Fs).';
            freqs = Fs*(0:(NFFT/2))/NFFT;
            freq_idx = find(abs((freqs-target_freq))==min(abs(freqs-target_freq)));

            magY = abs(fft_y);
            %phaseY = unwrap(angle(Y));
            phaseY = angle(fft_y);
            %phase_rad = atan2(imag(Y(freq_idx)), real(Y(freq_idx)));
            %phase_deg = phase_rad / pi * 180 + 90;
            
            % Store FFT analysis resuts to struct:
            % -------------------------------------------------------------
            fft_struct.(roi_no).targetPhase = phaseY(freq_idx); % unwrap(angle(Y(freq_idx)));
            fft_struct.(roi_no).targetMag = magY(freq_idx);
            fft_struct.(roi_no).fft_y = fft_y;
            fft_struct.(roi_no).DC_y = trace_y;
            fft_struct.(roi_no).raw_y = tmp0;
            fft_struct.(roi_no).slices = slice_indices;
            fft_struct.(roi_no).freqs = freqs;
            fft_struct.(roi_no).freq_idx = freq_idx;
            fft_struct.(roi_no).target_freq = target_freq;

            phase_map(masks(:,:,row)==1) = phaseY(freq_idx);
            mag_map(masks(:,:,row)==1) = magY(freq_idx);
            max_idx = find(magY==max(magY));
            max_map(masks(:,:,row)==1) = phaseY(max_idx(1));
            
            maps.magnitude = mag_map;
            maps.phase = phase_map;
            maps.max_map = max_map;
                            
            if strcamp(roiType, 'pixels')
                mapTypes = fieldnames(maps);
                for map=1:length(mapTypes)
                    currMap = maps.(mapTypes{map});
                    currMap = reshape(currMap, [d1, d2, size(currMap,3)]);
                    maps.(mapTypes{map}) = currMap;
                end
            end
            
        end
        M.file(fidx) = maps;
    end
    
    % Save maps for current slice:
    map_fn = sprintf('maps_Slice%02d', sidx);
    save(fullfile(figDir, map_fn), 'M', '-v7.3');
    
end