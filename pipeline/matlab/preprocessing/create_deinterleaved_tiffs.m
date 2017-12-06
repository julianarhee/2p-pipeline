function mcparams = create_deinterleaved_tiffs(A, mcparams)

% Takes "raw" (flyback-corrected) TIFFs from ./DATA and deinterleaves them into ./DATA/Parsed
% Assumes that mcparams.dest_dir = 'Parsed'

namingFunction = @defaultNamingFunction;

simeta = load(A.raw_simeta_path);

% No MC, so just parse raw tiffs in standard directory in tiff base dir:
%if ~exist(fullfile(mcparams.source_dir, mcparams.dest__dir), 'dir')
%    mkdir(fullfile(mcparams.source_dir, mcparams.dest_dir));
%end
%write_dir = fullfile(mcparams.source_dir, mcparams.dest_dir);

parse = true;
write_dir = fullfile(mcparams.source_dir, sprintf('%s_slices', mcparams.dest_dir));
if ~exist(write_dir, 'dir')
    mkdir(write_dir);
    parse = true;
else
    fprintf('Found existing write-dir: %s\n', write_dir);
    nchannel_dirs = dir(fullfile(write_dir, 'Channel*'));
    if length(nchannel_dirs)>0
        nfile_dirs = dir(fullfile(write_dir, nchannel_dirs(1).name, 'File*'));
        if length(nfile_dirs)>0
            nslices = dir(fullfile(write_dir, nchannel_dirs(1).name, nfile_dirs(1).name, '*.tif'));
            if length(nslices)==length(A.slices) && length(nfile_dirs)==A.ntiffs
                fprintf('Found correct number of deinterleaved tiffs in dir:\n')
                fprintf('%s\n', write_dir);
                user_says_parse = input('Press Y/n to re-deinterleave tiffs from above slice-dir: ', 's');
                if strcmp(user_says_parse, 'Y')
                    parse = true;
                else
                    fprintf('Parsed tiffs look good. Not redoing deinterleave step.\n');
                    parse = false;
                end
            end
        end
    else
        % may already be deinterleaved, not sorted into folders:
        nslices = dir(fullfile(write_dir, '*.tif'));
        if length(nslices)==A.ntiffs*A.nchannels*A.nvolumes
            fprintf('Found correct number of deinterleaved tiffs in dir:\n')
            fprintf('%s\n', write_dir);
            user_says_parse = input('Press Y/n to re-deinterleave tiffs from specified input: ', 's');
            if strcmp(user_says_parse, 'Y')
                parse = true;
            else
                fprintf('Parsed tiffs look good. Not redoing deinterleave step.\n');
                parse = false;
            end
        end 
    end
end

% Store empty method-specific info struct:
mcparams.info = struct();
mcparams.info.acquisition_name = A.base_filename;
mcparams.info.acq_object_path = '';

if parse 
    fprintf('Parsing RAW tiffs into %s\n', write_dir);
    % Parse TIFFs in ./functional/DATA (may be copies of raw tiffs, or flyback-corrected tiffs).
    sourcedir = mcparams.source_dir;
    tiffs_to_parse = dir(fullfile(sourcedir, '*.tif'));
    tiffs_to_parse = {tiffs_to_parse(:).name}';
    if length(tiffs_to_parse)==0
        while (1)
            fprintf('No raw tiffs found in specified source: %s\n', sourcedir);
            alt_source = input(sprintf('Checking child dir: %s\n. Press [ENTER to confirm], or enter child dir for alt source: \n', mcparams.dest_dir), 's');
            if length(alt_source)==0
                sourcedir = fullfile(mcparams.source_dir, mcparams.dest_dir);
                %mcparams.source_dir = sourcedir;
            else
                sourcedir = alt_source;
                %mcparams.source_dir = sourcedir;
            end
            tiffs_to_parse = dir(fullfile(sourcedir, '*.tif'));
            tiffs_to_parse = {tiffs_to_parse(:).name}';
            if length(tiffs_to_parse)>0
                break;
            end
        end
    end
    fprintf('Creating deinterleaved slice tiffs from source: %s\n', sourcedir);
    fprintf('Found %i original TIFFs to splice.\n', length(tiffs_to_parse));
           
    deinterleaved_folder = sprintf('%s_slices', mcparams.dest_dir);
    deinterleaved_dir = fullfile(mcparams.source_dir, deinterleaved_folder); 
    for fid=1:length(tiffs_to_parse)
        currtiffpath = fullfile(sourcedir, tiffs_to_parse{fid});
        currfile = sprintf('File%03d', fid);
%         if A.nchannels==1 && length(A.slices)==1
%             % Just move single-plane tiffs to standard file tree:
%             fprintf('Single channel, single slice. No need for deinterleaving.\n')
%             channelnum = simeta.(currfile).SI.hChannels.channelSave;
%             mov_filename = feval(namingFunction, A.base_filename, 1, channelnum, fid);
% 
%             currchannel = sprintf('Channel%02d', channelnum);
%             slicedir = fullfile(write_dir, currchannel, currfile);
%             if ~exist(slicedir, 'dir')
%                 mkdir(slicedir);
%             end
%             copyfile(currtiffpath, fullfile(slicedir, mov_filename)); 
%         else
%             if strfind(simeta.(currfile).SI.VERSION_MAJOR, '2016') 
%                 Y = read_file(currtiffpath);
%             else
%                 Y = read_imgdata(currtiffpath);
%             end
%             deinterleave_tiffs(Y, tiffs_to_parse{fid}, fid, write_dir, A);
%         end
%         
        if strfind(simeta.(currfile).SI.VERSION_MAJOR, '2016') 
            Y = read_file(currtiffpath);
        else
            Y = read_imgdata(currtiffpath);
        end
        deinterleave_tiffs(Y, tiffs_to_parse{fid}, fid, write_dir, A);

        fprintf('Finished deinterleaving %s of %s.\n', currfile, length(tiffs_to_parse));
    end
    fprintf('Finished parsing tiffs!\n');
end

% Move RAW tiffs to 'Raw' folder:
if ~exist(fullfile(mcparams.source_dir, 'Raw'), 'dir')
    mkdir(fullfile(mcparams.source_dir, 'Raw'));
end
tiffs_to_parse = dir(fullfile(mcparams.source_dir, '*.tif'));
tiffs_to_parse = {tiffs_to_parse(:).name}';
for fid=1:length(tiffs_to_parse)
    movefile(fullfile(mcparams.source_dir, tiffs_to_parse{fid}), fullfile(mcparams.source_dir, 'Raw', tiffs_to_parse{fid}));
end
fprintf('Moved %i raw tiff files into ./Data/Raw.\n', length(tiffs_to_parse));

end

function mov_filename = defaultNamingFunction(acqName, nSlice, nChannel, movNum)

mov_filename = sprintf('%s_Slice%02.0f_Channel%02.0f_File%03.0f.tif',...
    acqName, nSlice, nChannel, movNum);
end


% 
% % Add base filename if missing from ref struct:
% % For now, this is specific to Acquisition2P, since this uses the base filename to name field for acq obj.
% if ~isfield(A, 'base_filename') || ~strcmp(A.base_filename, mcparams.info.acquisition_name)
%     A.base_filename = mcparams.info.acquisition_name;
% end
% 
% 