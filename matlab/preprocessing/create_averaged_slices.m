function create_averaged_slices(mcparams)

% Grab (corrected) TIFFs from DATA (or acquisition) dir for correction:
data_files = dir(fullfile(mcparams.tiff_dir, '*.tif'));
fprintf('Creating averaged slice images for TIFFs in:\n%s\n', mcparams.tiff_dir);
data_dir = mcparams.tiff_dir; %D.dataDir;
data_files = {data_files(:).name}';

path_to_averages = mcparams.averaged_slices_dir;

for tiff_idx=1:length(data_files)
      
    for ch=1:mcparams.nchannels
        ch_path = fullfile(path_to_averages, sprintf('Channel%02d', ch), sprintf('File%03d', tiff_idx));
        ch_path_vis = fullfile(path_to_averages, sprintf('Channel%02d', ch), sprintf('File%03d_visible', tiff_idx));
        if ~exist(ch_path, 'dir')
            mkdir(ch_path)
            mkdir(ch_path_vis)
        end 
	fprintf('Averaging Channel %i, File %i...\n', ch, tiff_idx);
        if mcparams.correct_bidi
            slice_dir = fullfile(mcparams.tiff_dir, 'Corrected_Bidi', sprintf('Channel%02d', ch), sprintf('File%03d', tiff_idx))
        else
            slice_dir = fullfile(mcparams.tiff_dir, 'Corrected', sprintf('Channel%02d', ch), sprintf('File%03d', tiff_idx))
	end
        tiffs = dir(fullfile(slice_dir, '*.tif'));
        tiffs = {tiffs(:).name}'
        d3 = length(tiffs);
        sample = read_file(fullfile(slice_dir, tiffs{1}));
	d1=size(sample,1); d2=size(sample,2); clear sample
	
	avgs = zeros([d1,d2,d3]);
	for sl=1:d3
	    tiffdata = read_file(fullfile(slice_dir, tiffs{sl}));
	    fprintf('TIFF %i of %i: size is %s.\n', tiff_idx, length(tiffs), mat2str(size(tiffdata)));
	    avgs(:,:,sl) = mean(tiffdata, 3);
	    slicename = sprintf('average_Slice%02d_Channel%02d_File%03d.tif', sl, ch, tiff_idx);
	    tiffWrite(avgs(:,:,sl), slicename, ch_path);	
	end
	% make visible
	tmp = (avgs-min(avgs(:)))./(max(avgs(:))-min(avgs(:)));
	for sl=1:d3       
            avgs_visible = adapthisteq(tmp(:,:,sl));
	    slicename_vis = sprintf('average_Slice%02d_Channel%02d_File%03d_vis.tif', sl, ch, tiff_idx);
	    tiffWrite(avgs_visible*((2^16)-1), slicename_vis, ch_path_vis);
        end
    end
end
end
