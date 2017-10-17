function extract_traces(I, A)

% roiparams.params 
% roiparams.nrois
% roiparams.maskpaths 
% roiparams.maskpath3D
% roiparams.sourcepaths
% roiparams.roi_info

mcparams = load(A.mcparams_path);
mcparams = mcparams.(I.mc_id);

roiparams_path = fullfile(A.roi_dir, I.roi_id, 'roiparams.mat');
%load(A.roiparams_path);
load(roiparams_path);
simeta = load(A.raw_simeta_path);

% Set path for slice time-series tiffs:
if I.use_bidi_corrected
    base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir);
else
    if mcparams.corrected
        base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.corrected_dir);
    else
        base_slice_dir = fullfile(mcparams.tiff_dir, mcparams.parsed_dir);
    end
end

switch I.roi_method
    
    case 'pixels'
        % do sth
        
    case 'manual2D'
        % do stuff
        
    case 'pyblob2D'
        % Load roistruct created in roi_blob_detector.py:
        for sidx = 1:length(I.slices)
            sl = I.slices(sidx);
            load(roiparams.maskpaths{sidx});
            maskcell = arrayfun(@(roi) make_sparse_masks(masks(:,:,roi)), 1:size(masks,3), 'UniformOutput', false);
            maskcell = cellfun(@logical, maskcell, 'UniformOutput', false);
             
            % load time-series for current slice:
            for fidx=1:A.ntiffs
                curr_file_path = fullfile(base_slice_dir, sprintf('Channel%02d', I.signal_channel), sprintf('File%03d', fidx));
                % TODO: adjust so that path to slice tiff is not dependent on mcparams.info.acquisition_name
                % since this is currently specific to mcparams.method=Acquisition2P
                curr_file = sprintf('%s_Slice%02d_Channel%02d_File%03d.tif', mcparams.info.acquisition_name, sl, I.signal_channel, fidx)
                %Y = tiffRead(fullfile(curr_file_path, curr_file));
                
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
            save(fullfile(A.trace_dir, I.roi_id, tracestruct_name), '-struct', 'tracestruct');
        end
        
        
    case 'pyblob3D'
        % do similar stuff
        
    case 'manual3D'
        % do other stuff
        
    case 'cnmf3D'
    
    
    
end
