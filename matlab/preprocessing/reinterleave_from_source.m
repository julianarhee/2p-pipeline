function Yr = reinterleave_from_source(filedir, sourcedir, A)

channel_dirs = dir(sourcedir);
channel_dirs = channel_dirs(arrayfun(@(x) ~strcmp(x.name(1), '.'), channel_dirs));  % remove hidden
channel_dirs = {channel_dirs(:).name}';

nslices = length(A.slices);
nchannels = A.nchannels;
nfiles = A.ntiffs;
nvolumes = A.nvolumes;
nframes = nslices*nvolumes*nchannels;
d1=A.lines_per_frame; d2=A.pixels_per_line;

sliceidxs = 1:nchannels:nslices*nchannels;

Yr = zeros(d1, d2, nframes);

for ch=1:length(channel_dirs)
    tiff_slice_fns = dir(fullfile(sourcedir, channel_dirs{ch}, filedir, '*.tif'));
    tiff_slice_fns = {tiff_slice_fns(:).name}'
 
    for sl=1:length(tiff_slice_fns)
        
         %curr_slice = read_file(fullfile(sourcedir, channel_dirs{ch}, filedir, tiff_slice_fns{sl}));
         currtiffpath = fullfile(sourcedir, channel_dirs{ch}, filedir, tiff_slice_fns{sl});
         curr_file_name = sprintf('File%03d', fidx);
         if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
             curr_slice = read_file(currtiffpath);
         else
             curr_slice = read_imgdata(currtiffpath);
         end  
         if ch==1
             Yr(:,:,sliceidxs(sl):(nslices*nchannels):end) = curr_slice;
         else
             Yr(:,:,(sliceidxs(sl)+1):(nslices*nchannels):end) = curr_slice;
         end
    end

end
