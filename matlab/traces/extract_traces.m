function extract_traces(I, mcparams, A)

% roiparams.params 
% roiparams.nrois
% roiparams.maskpaths 
% roiparams.maskpath3D
% roiparams.sourcepaths
% roiparams.roi_info

%mcparams = load(A.mcparams_path);
%mcparams = mcparams.(I.mc_id);

roiparams_path = fullfile(A.roi_dir, I.roi_id, 'roiparams.mat');
roiparams = load(roiparams_path);
if ismember('roiparams', fieldnames(roiparams))
    roiparams = roiparams.roiparams;
end
%roiparams = load(roiparams_path)
%roiparams = roiparams.(I.roi_id)

simeta = load(A.raw_simeta_path);

% Set path for slice time-series tiffs:
% if I.use_bidi_corrected
%     base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
% else
%     if mcparams.corrected
%         base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.corrected_dir);
%     else
%         base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.parsed_dir);
%     end
% end

% base_slice_dir = fullfile(mcparams.source_dir, sprintf('%s_slices', mcparams.dest_dir));

curr_tracestruct_dir = fullfile(A.trace_dir, A.trace_id.(I.analysis_id));
fprintf('Saving tracestructs to: %s\n', curr_tracestruct_dir);

% Load roistruct created in roi_blob_detector.py:
for sidx = 1:length(I.slices)
    sl = I.slices(sidx);
    load(roiparams.maskpaths{sidx});
    maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
    maskcell = cellfun(@logical, maskcell, 'UniformOutput', false);
     
    % load time-series for current slice:
    for fidx=1:A.ntiffs
        if length(A.slices)>1 || A.nchannels>1
            base_slice_dir = fullfile(mcparams.source_dir, sprintf('%s_slices', mcparams.dest_dir)); 
            curr_file_path = fullfile(base_slice_dir, sprintf('Channel%02d', I.signal_channel), sprintf('File%03d', fidx));
            curr_file = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', A.base_filename, sl, I.signal_channel, fidx)
        else
            base_slice_dir = fullfile(mcparams.source_dir, sprintf('%s', mcparams.dest_dir));
            curr_file_path = base_slice_dir;
            tmpfiles = dir(fullfile(curr_file_path, '*.tif'));
            tmpfiles = {tmpfiles(:).name}'
            curr_file = tmpfiles{fidx};
        end 
        currtiffpath = fullfile(curr_file_path, curr_file);
        curr_file_name = sprintf('File%03d', fidx);
        if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
            Y = read_file(currtiffpath);
        else
            Y = read_imgdata(currtiffpath);
        end 

        % TODO: add option to check for MC-evalation for
        % interpolating frames.
        
        % Get raw traces:
        rawtracemat = get_raw_traces(Y, maskcell);
        
        tracestruct.file(fidx).rawtracemat = rawtracemat;
        tracestruct.file(fidx).maskcell = maskcell;
        tracestruct.file(fidx).maskpath = roiparams.maskpaths{sidx};
        tracestruct.file(fidx).nrois = size(masks,3);
    end
    tracestruct_name = sprintf('traces_Slice%02d_Channel%02d.mat', sl, I.signal_channel);
    save(fullfile(curr_tracestruct_dir, tracestruct_name), '-struct', 'tracestruct');

% 
% switch I.roi_method
%     
%     case 'pixels'
%         % do sth
%         
%     case 'manual2D'
%         % Tested on 20171009_CE059 data - runs for gratings_phaseMod using pyblob3D code
% 
%         
%     case 'pyblob2D'
%         % Load roistruct created in roi_blob_detector.py:
%         for sidx = 1:length(I.slices)
%             sl = I.slices(sidx);
%             load(roiparams.maskpaths{sidx});
%             maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
%             maskcell = cellfun(@logical, maskcell, 'UniformOutput', false);
%              
%             % load time-series for current slice:
%             for fidx=1:A.ntiffs
%                 curr_file_path = fullfile(base_slice_dir, sprintf('Channel%02d', I.signal_channel), sprintf('File%03d', fidx));
%                 curr_file = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', A.base_filename, sl, I.signal_channel, fidx)
%                 
%                 currtiffpath = fullfile(curr_file_path, curr_file);
%                 curr_file_name = sprintf('File%03d', fidx);
%                 if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
%                     Y = read_file(currtiffpath);
%                 else
%                     Y = read_imgdata(currtiffpath);
%                 end 
%  
%                 % TODO: add option to check for MC-evalation for
%                 % interpolating frames.
%                 
%                 % Get raw traces:
%                 rawtracemat = get_raw_traces(Y, maskcell);
%                 
%                 tracestruct.file(fidx).rawtracemat = rawtracemat;
%                 tracestruct.file(fidx).maskcell = maskcell;
%                 tracestruct.file(fidx).maskpath = roiparams.maskpaths{sidx};
%                 tracestruct.file(fidx).nrois = size(masks,3);
%             end
%             tracestruct_name = sprintf('traces_Slice%02d_Channel%02d.mat', sl, I.signal_channel);
%             save(fullfile(curr_tracestruct_dir, tracestruct_name), '-struct', 'tracestruct');
%         end
%         
%         
%     case 'pyblob3D'
%         % do similar stuff
%         
%     case 'manual3D'
%         % do other stuff
%         
%     case 'cnmf3D'
%     
%     
    
end
