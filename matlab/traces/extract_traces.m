function extract_traces(A)

% roiparams.params 
% roiparams.nrois
% roiparams.maskpaths 
% roiparams.maskpath3D
% roiparams.sourcepaths
% roiparams.roi_info

load(A.mcparams_path);
load(A.roiparams_path);

% Set path for slice time-series tiffs:
if A.use_bidi_corrected
    base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
else
    if mcparams.corrected
        base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.corrected_dir);
    else
        base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.parsed_dir);
    end
end

switch A.roi_method
    
    case 'pixels'
        % do sth
        
    case 'manual2D'
        % do stuff
        
    case 'pyblob2D'
        % Load roistruct created in roi_blob_detector.py:
        for sidx = 1:length(A.slices)
            sl = A.slices(sidx);
            load(roiparams.maskpaths{sidx});
            maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
            maskcell = cellfun(@logical, maskcell, 'UniformOutput', false);
             
            % load time-series for current slice:
            for fidx=1:A.ntiffs
                curr_file_path = fullfile(base_slice_dir, sprintf('Channel%02d', A.signal_channel), sprintf('File%03d', fidx));
                % TODO: adjust so that path to slice tiff is not dependent on mcparams.info.acquisition_name
                % since this is currently specific to mcparams.method=Acquisition2P
                curr_file = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', mcparams.info.acquisition_name, sl, A.signal_channel, fidx)
                Y = tiffRead(fullfile(curr_file_path, curr_file));
                
                % TODO: add option to check for MC-evalation for
                % interpolating frames.
                
                % Get raw traces:
                rawtracemat = get_raw_traces(Y, maskcell);
                
                tracestruct.file(fidx).rawtracemat = rawtracemat;
                tracestruct.file(fidx).maskcell = maskcell;
                tracestruct.file(fidx).maskpath = roiparams.maskpaths{sidx};
                tracestruct.file(fidx).nrois = size(masks,3);
            end
            tracestruct_name = sprintf('traces_Slice%02d_Channel%02d.mat', sl, A.signal_channel);
            save(fullfile(A.trace_dir, tracestruct_name), '-struct', 'tracestruct');
        end
        
        
    case 'pyblob3D'
        % do similar stuff
        
    case 'manual3D'
        % do other stuff
        
    case 'cnmf3D'
    
    
    
end
