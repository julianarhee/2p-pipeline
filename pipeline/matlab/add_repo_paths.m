function add_repo_paths(varargin)
%% Set paths
if length(varargin) == 0
    cvx_path = '~/MATLAB/cvx';
    repo_prefix = '~/Repositories';
elseif length(varargin) == 1
    cvx_path = varargin{1}
    repo_prefix = '~/Repositories'
elseif length(varargin) == 2
    cvx_path = varargin{1}
    repo_prefix = varargin{2}
end


%addpath(genpath(fullfile(repo_prefix, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_prefix, 'NoRMCorre')));
addpath(genpath(fullfile(repo_prefix, 'Acquisition2P_class')));
addpath(genpath(fullfile(repo_prefix, 'helperFunctions')));
addpath(genpath(fullfile(repo_prefix, '12k2p-software')));

if exist(cvx_path, 'dir')
    cd(cvx_path);
    cvx_setup;
end
addpath(genpath(fullfile(repo_prefix, '2p-pipeline')));
pipe_dir = fullfile(repo_prefix, '2p-pipeline', 'pipeline');

cd(pipe_dir)
 
end
