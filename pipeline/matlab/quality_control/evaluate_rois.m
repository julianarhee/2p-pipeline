function evaluate_rois(options)


    %unpack options
    if isfield(options,'get_pixel_corr')
        get_pixel_corr = options.get_pixel_corr;
    else
        get_pixel_corr = 0;
    end

    if isfield(options,'get_trial_corr')
        get_trial_corr = options.get_trial_corr;
    else
        get_trial_corr = 0;
    end

    if isfield(options,'get_file_corr')
        get_file_corr = options.get_file_corr;
    else
        get_file_corr = 0;
    end



    %load reference file
    path_to_reference = fullfile(options.acquisition_base_dir, sprintf('reference_%s.mat', options.tiff_source));
    A = load(path_to_reference);
    if ~isfield(A,'roi_dir')
        A.roi_dir = fullfile(options.acquisition_base_dir,'ROIs');
    end

    %load mcparams
    mcparams = load(fullfile(options.data_dir,'mcparams.mat'));
    %retrieve relevant mcparams sub-structure
    mcparams = mcparams.(options.mcparams_id);

    %load roiparams
    roiparams_path = fullfile(A.roi_dir, options.roi_folder, 'roiparams.mat');
    roiparams = load(roiparams_path);
    if isfield(roiparams,'roiparams')
        roiparams = roiparams.roiparams;
    end

    %load simeta
    simeta = load(A.raw_simeta_path);

    base_slice_dir = fullfile(mcparams.source_dir,sprintf('%s_slices',mcparams.dest_dir));%*NOTE: hard-coding now, but should come from mcparams

    if ~isdir(fullfile(A.roi_dir, options.roi_folder, 'metrics'))
        mkdir(fullfile(A.roi_dir, options.roi_folder, 'metrics'))
    end

    % Load roistruct created in roi_blob_detector.py:
    for sidx = 1:length(options.sourceslices)
        sl = options.sourceslices(sidx);

        %load masks
        load(roiparams.maskpaths{sl}); 
        maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
        maskcell = cellfun(@logical, maskcell, 'UniformOutput', false);


        %initialize strctures
        if get_pixel_corr
            pixcorrstruct = struct;
            pixcorrstruct.options = options;
            pixcorrstruct.maskpath = roiparams.maskpaths{sl};
        end
        if get_file_corr
            filecorrstruct = struct;
            filecorrstruct.options = options;
            filecorrstruct.maskpath = roiparams.maskpaths{sl};
        end
        if get_trial_corr
            trialcorrstruct = struct;
            trialorrstruct.options = options;
            trialcorrstruct.maskpath = roiparams.maskpaths{sl};
        end

        % load time-series for current slice:
        for fidx=1:A.ntiffs
            curr_file_path = fullfile(base_slice_dir, sprintf('Channel%02d', A.signal_channel.(options.analysis_id)), sprintf('File%03d', fidx));

            curr_file = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', A.base_filename,  sl, A.signal_channel.(options.analysis_id), fidx)

            currtiffpath = fullfile(curr_file_path, curr_file);
            curr_file_name = sprintf('File%03d', fidx);
            if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
                Y = read_file(currtiffpath);
            else
                Y = read_imgdata(currtiffpath);
            end 

            if get_pixel_corr
                pixcorrstruct.file(fidx).sourcefile = curr_file;
                % Get mean pixel-to-pixel time correlation for each ROI
                mean_pixel_corr = get_pixel_time_correlation(maskcell,Y);
                pixcorrstruct.file(fidx).mean_pixel_corr = mean_pixel_corr;
            end


        end

        if get_pixel_corr
            %save pixel correlation
            pixcorrstruct_name = sprintf('Slice%02d_Channel%02d_pixcorrstruct.mat', sl, A.signal_channel.(options.analysis_id));
            save(fullfile(A.roi_dir, options.roi_folder, 'metrics', pixcorrstruct_name), '-struct', 'pixcorrstruct');
        end

        if get_file_corr
            % Get mean file-to-file correlation for each roi
            % *Note: only valid if stimulus sequence was 
            % the same across files

            %load trace file
            tracedata_dir = fullfile(A.trace_dir, options.roi_folder,options.mcparams_id);
            tracedata_file = sprintf('traces_Slice%02d_Channel%02d.mat',sl, A.signal_channel.(options.analysis_id));
            tracestruct = load(fullfile(tracedata_dir,tracedata_file));
            filecorrstruct.mean_file_corr = get_file_correlation(tracestruct);

            filename = sprintf('Slice%02d_Channel%02d_filecorrstruct.mat', sl, A.signal_channel.(options.analysis_id));
            save(fullfile(A.roi_dir,options.roi_folder, 'metrics',filename),'-struct', 'filecorrstruct');
        end

        if get_trial_corr
            %get trial data
            trialdata_dir = fullfile(A.trace_dir, options.roi_folder,options.mcparams_id,'Parsed');
            trialdata_file = sprintf('stimtraces_Channel%02d_Slice%02d.mat', A.signal_channel.(options.analysis_id), sl);
            trialstruct = load(fullfile(trialdata_dir,trialdata_file));
            trialcorrstruct.sourcefile = fullfile(trialdata_dir,trialdata_file);

            %get trial-to-trial correlation
            trialcorrstruct.mean_trial_corr = get_trial_correlation(trialstruct);


            %save trial-to-trial correlation
            filename = sprintf('Slice%02d_Channel%02d_trialcorrstruct', sl, A.signal_channel.(options.analysis_id));
            save(fullfile(A.roi_dir,options.roi_folder, 'metrics',filename), '-struct', 'trialcorrstruct');
        end
    end
end