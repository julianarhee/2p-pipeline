function bash_process_tiffs(acquisition_dir, varargin)

% This function should be used after running:
%
% init_header;
% check_init;
% initialize_analysis;

% This are currently interacive steps to make sure nothing is overwritten.
% The last step produces a tmp file in acquisition_dir that stores relevant vars for process_tiffs.

if length(varargin)==0
    functional = 'functional';
else
    functional = varargin{1};
end

A = load(fullfile(acquisition_dir, sprintf('reference_%s.mat', functional)));

tmpvars = load(fullfile(acquisition_dir, 'tmp_init_variables.mat'));
if ismember('tmpvars', fieldnames(tmpvars))
    tmpvars = tmpvars.tmpvars;
end

I = tmpvars.I;

new_mc = tmpvars.new_mc;

clear tmpvars

process_tiffs(I, A, new_mc);


