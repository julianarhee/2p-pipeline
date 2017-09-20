function sort_deinterleaved_tiffs(mcparams)

corrected_path = mcparams.output_dir;
nchannels = mcparams.nchannels;

if ~exist('namingFunction', 'var')
    namingFunction = @defaultNamingFunction;
end

% ---------------------------------------------------------------------
% If using (and including) 2 channels for MC, separate them into their
% own dirs:
fprintf('Moving files...\n');

%corrected_path = fullfile(acquisition_dir, tiff_dir);
corrected_tiff_fns = dir(fullfile(corrected_path, '*.tif'));
corrected_tiff_fns = {corrected_tiff_fns(:).name};
corrected_ch1_path = fullfile(corrected_path, 'Channel01');
if nchannels == 2
    corrected_ch2_path = fullfile(corrected_path, 'Channel02');
end
if ~exist(corrected_ch1_path, 'dir')
    mkdir(corrected_ch1_path);
    if nchannels==2
        mkdir(corrected_ch2_path);
    end
end
for tiff_idx=1:length(corrected_tiff_fns)
    if strfind(corrected_tiff_fns{tiff_idx}, 'Channel01')
        movefile(fullfile(corrected_path, corrected_tiff_fns{tiff_idx}), fullfile(corrected_ch1_path, corrected_tiff_fns{tiff_idx}));
    else
        movefile(fullfile(corrected_path, corrected_tiff_fns{tiff_idx}), fullfile(corrected_ch2_path, corrected_tiff_fns{tiff_idx}));
    end
end 


% ---------------------------------------------------------------------
% If multiple files/runs of a given acqusition (i.e., FOV), separate files:
files_found = {};
channel_dirs = dir(corrected_path);
%csub = [channel_dirs(:).isdir];
channel_dirs = channel_dirs(arrayfun(@(x) ~strcmp(x.name(1),'.'),channel_dirs));
channels = {channel_dirs(:).name}';   
for cidx=1:length(channels)
    channel_path = fullfile(corrected_path, channels{cidx});
    channel_tiffs = dir(fullfile(channel_path, '*.tif'));
    channel_tiffs = {channel_tiffs(:).name}';
    if isempty(channel_tiffs)
        % check naming:
        file_paths = dir(fullfile(channel_path, 'File*'));
        file_paths = {file_paths(:).name}';
        for fpath = 1:length(file_paths)
            tiffs = dir(fullfile(channel_path, file_paths{fpath}, '*.tif'));
            tiffs = {tiffs(:).name}';
            if isempty(strfind(tiffs{1}, sprintf('File%03d', fpath)))
                % Rename files:
                for tidx=1:length(tiffs)
                    % Fiji-split:
                    splits = regexp(tiffs{tidx}, '\s', 'split');
                    nChannel = str2double(splits{2}(regexp(splits{2}, '\d')));
                    nSlice = str2double(splits{3}(regexp(splits{3}, '\d')));
                    new_file_name = feval(namingFunction, mcparams.acquisition_name, nSlice, nChannel, fpath);
                    movefile(fullfile(channel_path, file_paths{fpath}, tiffs{tidx}), fullfile(channel_path, file_paths{fpath}, new_file_name));
                end
            end
        end
    else
        % Sort parsed channel tiffs into FILE dirs:
        for tiff_idx=1:length(channel_tiffs)
            file_parts = strsplit(channel_tiffs{tiff_idx}, 'File');
            if length(files_found)==0
                files_found{1} = cellstr(file_parts(end));
            else
                files_found{end+1} = cellstr(file_parts(end));
            end
        end
        nfiles = unique([files_found{:}]);
        if length(nfiles) > 1
            for fidx=1:length(nfiles)
                %corrected_file_path = sprintf('%s_Channel%02d_File%03d', tiff_dir, cidx, fidx);
                corrected_file_path = sprintf('File%03d', fidx);
                if ~exist(fullfile(channel_path, corrected_file_path), 'dir')
                    mkdir(fullfile(channel_path, corrected_file_path));
                end

                for tiff_idx=1:length(channel_tiffs)
                    if strfind(channel_tiffs{tiff_idx}, nfiles{fidx})
                        movefile(fullfile(channel_path, channel_tiffs{tiff_idx}), fullfile(channel_path, corrected_file_path, channel_tiffs{tiff_idx}));
                    end
                end
            end
        end
    end
end

% end
       


end

function movFileName = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

movFileName = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end