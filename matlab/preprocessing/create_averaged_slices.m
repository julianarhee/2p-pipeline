function create_averaged_slices(mcparams)

% Grab (corrected) TIFFs from DATA (or acquisition) dir for correction:
tiffs = dir(fullfile(mcparams.tiff_dir, '*.tif'));
fprintf('Creating averaged slice images for TIFFs in:\n%s\n', mcparams.tiff_dir);
tiff_dir = mcparams.tiff_dir; %D.dataDir;
tiffs = {tiffs(:).name}';

path_to_averages = mcparams.averaged_slices_dir;

for tiff_idx=1:length(tiffs)
    tpath = fullfile(mcparams.tiff_dir, tiffs{tiff_idx});
    tiffdata = read_file(tpath);
    d1=size(tiffdata,1); d2=size(tiffdata,2); d3=size(tiffdata,3);
    fprintf('TIFF %i of %i: size is %s.\n', tiff_idx, length(tiffs), mat2str(size(tiffdata)));
    
    for ch=1:mcparams.nchannels
        ch_path = fullfile(path_to_averages, sprintf('Channel%02d', ch), sprintf('File%03d', tiff_idx));
        ch_path_vis = fullfile(path_to_averages, sprintf('Channel%02d', ch), sprintf('File%03d_visible', tiff_idx));
        if ~exist(ch_path, 'dir')
            mkdir(ch_path)
            mkdir(ch_path_vis)
        end 
	fprintf('Averaging Channel %i, File %i...\n', ch, tiff_idx);

	avgs = zeros([d1,d2,d3]);
	for sl=1:d3
	    if strfind(tiffs{tiff_idx}, sprintf('_Channel%02d', ch))    % channels are split
		avgs(:,:,sl) = mean(tiffdata(:,:,sl,:), 4);
	    else
		avgs(:,:,sl) = mean(tiffdata(:,:,sl,ch:mcparams.nchannels:end), 4);
	    end
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
