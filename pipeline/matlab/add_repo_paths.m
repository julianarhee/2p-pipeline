function add_repo_paths(vargin)
%% Set paths
if length(vargin) == 0
    repo_prefix = '~/Repositories';
else
    repo_prefix = varargin{1};
end

addpath(genpath(fullfile(repo_prefix, '2p-pipeline')));
addpath(genpath(fullfile(repo_prefix, 'ca_source_extraction')));
addpath(genpath(fullfile(repo_prefix, 'NoRMCorre')));
addpath(genpath(fullfile(repo_prefix, 'Acquisition2P_class')));
addpath(genpath(fullfile(repo_prefix, 'helperFunctions')));
addpath(genpath(fullfile(repo_prefix, '12k2p-software')));
if exist(fullfile(repo_prefix, '2p-pipeline', 'pipeline', 'matlab', 'helperfuncs', 'cvx'), 'dir')
    cd fullfile(repo_prefix, '2p-pipeline', 'pipeline', 'matlab', 'helperfuncs', 'cvs'); cvx_setup;
end
cd fullfile(repo_prefix, '2p-pipeline', 'pipeline')


