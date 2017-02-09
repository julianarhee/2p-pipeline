
nslices = 20;
start_slice = 5;
channel = 1;

source_dir = '/nas/volume1/2photon/RESDATA/20161218_CE024_highres/acquisitions/';
acquisition_dirs = dir(source_dir);
isub = [acquisition_dirs(:).isdir]; %# returns logical vector
acquisitions = {acquisition_dirs(isub).name}';
acquisitions(ismember(acquisitions,{'.','..'})) = [];

for acquisition_idx = 14:14 %1:length(acquisitions)
    
    curr_acquisition_name = acquisitions{acquisition_idx};
    curr_acquisition_dir = fullfile(source_dir, curr_acquisition_name);
    
    fprintf('Processing acquisition %s...\n', curr_acquisition_name);
    
    subvolume_dirs = dir(curr_acquisition_dir);
    isub = [subvolume_dirs(:).isdir]; %# returns logical vector
    subvolumes = {subvolume_dirs(isub).name}';
    subvolumes(ismember(subvolumes,{'.','..'})) = [];
    
    for subvolume_idx=1:length(subvolumes)
        if channel==1
            curr_subvolume_dir = fullfile(curr_acquisition_dir, subvolumes{subvolume_idx}, 'Corrected', 'Channel01');
            avg_channel_dir = 'average_stacks';
        else
            curr_subvolume_dir = fullfile(curr_acquisition_dir, subvolumes{subvolume_idx}, 'Corrected', 'Channel02');
            avg_channel_dir = 'average_stacks_BV';
        end

        slices = dir(fullfile(curr_subvolume_dir, '*.tif'));
        slices = {slices(:).name}';
        slices = slices(start_slice:end);
        
        average_stack = zeros(512,512,nslices-start_slice+1);
        for slice_idx=1:length(slices)
            curr_slice_path = fullfile(curr_subvolume_dir, slices{slice_idx});
            
            [Y,~] = tiffRead(curr_slice_path);
            %[header,Aout,imgInfo] = scanimage.util.opentif(curr_slice_path);
            sframe = 1;
            %Y = bigread2(curr_slice_path,sframe);
            %if ~isa(Y,'double');    Y = double(Y)./65535;  end         % convert to single

            [d1,d2,T] = size(Y);                                % dimensions of dataset
            d = d1*d2;  
            
            average_stack(:,:,slice_idx) = mean(Y, 3);
        end
        %average_stack = uint16(round(average_stack.*65535));
        %average_stack = int16(average_stack);
        avgerage_stack_dir = fullfile(curr_acquisition_dir, avg_channel_dir);
        if ~exist(avgerage_stack_dir, 'dir')
            mkdir(avgerage_stack_dir);
        end
        average_stack_fn = sprintf('avg_%s_sub%i_%i.tif', subvolumes{subvolume_idx}, start_slice, nslices);
        try
            tiffWrite(average_stack, average_stack_fn, avgerage_stack_dir);
        catch
            pause(60);
            tiffWrite(average_stack, average_stack_fn, avgerage_stack_dir) %, 'int16');
        end
            
    end
    
end
