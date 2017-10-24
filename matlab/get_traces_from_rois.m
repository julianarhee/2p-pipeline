function get_traces_from_rois(I, A)

%% Specify ROI param struct path:
% if get_rois_and_traces

% Load curr_mcparams:
mcparams = load(A.mcparams_path);
curr_mcparams = mcparams.(I.mc_id);


%% Specify Traces param struct path:
curr_trace_dir = fullfile(A.trace_dir, A.trace_id.(I.analysis_id));
if ~exist(curr_trace_dir)
    mkdir(curr_trace_dir)
end

%% Get traces
fprintf('Extracting RAW traces.\n')
extract_traces(I, curr_mcparams, A);
fprintf('Extracted raw traces.\n')

%% GET metadata for SI tiffs:

si = get_scan_info(I, A)
save(A.simeta_path.(I.analysis_id), '-struct', 'si');


%% Process traces
fprintf('Doing rolling mean subtraction.\n')

% For retino-movie:
% targetFreq = meta.file(1).mw.targetFreq;
% winUnit = (1/targetFreq);
% crop = meta.file(1).mw.nTrueFrames; %round((1/targetFreq)*ncycles*Fs);
% nWinUnits = 3;

% For PSTH:
win_unit = 6; %3;   % size of one window (sec)
num_units = 8; %3;  % number of win_units that make up sliding window 

tracestruct_names = get_processed_traces(I, A, win_unit, num_units);

fprintf('Done processing Traces!\n');

%% Get df/f for full movie:
fprintf('Getting df/f for file.\n');

df_min = 50;

get_df_traces(I, A, df_min);



fprintf('Extracted all traces for analysis ID %s | ROI ID: %s | MC ID: %s.\n', I.analysis_id, I.roi_id, I.mc_id);

% save(path_to_reference, '-struct', 'A', '-append')
% 
% % Also save json:
% savejson('', A, path_to_reference_json);
% 
% fprintf('DONE!\n');
% 

end
