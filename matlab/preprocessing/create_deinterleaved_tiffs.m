function [A, mcparams] = create_deinterleaved_tiffs(A, mcparams)

simeta = load(A.raw_simeta_path);

% No MC, so just parse raw tiffs in standard directory in tiff base dir:
mcparams.parsed_dir = 'Parsed';
if ~exist(fullfile(mcparams.tiff_dir, mcparams.parsed_dir), 'dir')
    mkdir(fullfile(mcparams.tiff_dir, mcparams.parsed_dir));
end

% TODO:  add plain raw tiff parsing here.
% notes:  need to account for flyback-correction check

% Store empty method-specific info struct:
mcparams.info = struct();
mcparams.info.acquisition_name = A.base_filename;
mcparams.info.ac_object_path = '';

% Parse TIFFs in ./functional/DATA (may be copies of raw tiffs, or flyback-corrected tiffs).
tiffs_to_parse = dir(fullfile(A.data_dir, '*.tif'));
tiffs_to_parse = {tiffs_to_parse(:).name}';
deinterleaved_dir = fullfile(mcparams.tiff_dir, mcparams.parsed_dir); 
for fid=1:length(tiffs_to_parse)
    %Y = read_file(fullfile(A.data_dir, tiffs_to_parse{fid}));
    currtiffpath = fullfile(A.data_dir, tiffs_to_parse{fid});
    curr_file_name = sprintf('File%03d', fidx);
    if strfind(simeta.(curr_file_name).SI.VERSION_MAJOR, '2016') 
        Y = read_file(currtiffpath);
    else
        Y = read_imgdata(currtiffpath);
    end
    deinterleave_tiffs(Y, tiffs_to_parse{fid}, fid, write_dir, A);
end

save(fullfile(mcparams.tiff_dir, 'mcparams.mat'), 'mcparams', '-append');
fprintf('Finished parsing tiffs!\n');

% Add base filename if missing from ref struct:
% For now, this is specific to Acquisition2P, since this uses the base filename to name field for acq obj.
if ~isfield(A, 'base_filename') || ~strcmp(A.base_filename, mcparams.info.acquisition_name)
    A.base_filename = mcparams.info.acquisition_name;
end


