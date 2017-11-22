function mcmetrics = evaluate_correction(I, A)

min_corr = 0.6

mcparams = load(A.mcparams_path);
mcparams = mcparams.(I.mc_id);

% For now, just use averaged time-series to check correction against ref:
average_slices_dir = fullfile(A.data_dir.(I.analysis_id), sprintf('Averaged_Slices_%s', I.average_source));
ch_dir = fullfile(average_slices_dir, sprintf('Channel%02d', mcparams.ref_channel));

ref_img_fn = dir(fullfile(ch_dir, sprintf('File%03d', mcparams.ref_file), '*.tif'));
ref_img_fn = ref_img_fn.name;

ref_img_path = fullfile(ch_dir, sprintf('File%03d', mcparams.ref_file), ref_img_fn);
ref_img = imread(ref_img_path);

corrvals = zeros(1, A.ntiffs);
for fidx=1:A.ntiffs
    curr_img_fn = dir(fullfile(ch_dir, sprintf('File%03d', fidx), '*.tif'));
    curr_img_fn = curr_img_fn.name;
    curr_img_path = fullfile(ch_dir, sprintf('File%03d', fidx), curr_img_fn);
    curr_img = imread(curr_img_path);
    
    corrvals(fidx) = corr2(ref_img, curr_img);
end
 
bad_files = find(corrvals<min_corr);

%mcmetrics_path = fullfile(A.data_dir.(I.analysis_id), sprintf('metrics_%s.mat', I.mc_id));
%mcmetrics_path_json = fullfile(A.data_dir.(I.analysis_id), sprintf('metrics_%s.json', I.mc_id));
mcmetrics_path = fullfile(A.data_dir.(I.analysis_id), 'mcmetrics.mat');
mcmetrics_path_json = fullfile(A.data_dir.(I.analysis_id),'mcmetrics.json');


if ~exist(mcmetrics_path, 'file')
    mcmetrics = struct()
    mcmetrics.(I.mc_id) = struct();
else
    mcmetrics = load(fullfile(mcmetrics_path));
end
mcmetrics.(I.mc_id).analysis_id = I.analysis_id;
mcmetrics.(I.mc_id).min_corr = min_corr;
mcmetrics.(I.mc_id).bad_files = bad_files

save(mcmetrics_path, '-struct', 'mcmetrics');

savejson('', mcmetrics, mcmetrics_path_json);

end
