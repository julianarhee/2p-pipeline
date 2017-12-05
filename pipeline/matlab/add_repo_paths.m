function add_repo_paths(varargin)
%% Set paths
if length(varargin) == 0
    repo_prefix = '~/Repositories';
else
    repo_prefix = varargin{1}
end


%addpath(genpath(fullfile(repo_prefix, 'ca_source_extraction')));
%addpath(genpath(fullfile(repo_prefix, 'NoRMCorre')));
addpath(genpath(fullfile(repo_prefix, 'Acquisition2P_class')));
addpath(genpath(fullfile(repo_prefix, 'helperFunctions')));
%addpath(genpath(fullfile(repo_prefix, '12k2p-software')));

cvx_dir = fullfile(repo_prefix, '2p-pipeline', 'pipeline', 'matlab', 'helperfuncs', 'cvx')
if exist(cvx_dir, 'dir')
    cd(cvx_dir);
    cvx_setup;
end
addpath(genpath(fullfile(repo_prefix, '2p-pipeline')));
pipe_dir = fullfile(repo_prefix, '2p-pipeline', 'pipeline');

cd(pipe_dir)
 
end
