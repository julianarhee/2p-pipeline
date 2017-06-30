%% Set paths
addpath(genpath('~/Repositories/2p-tester-scripts'));
addpath(genpath('~/Repositories/ca_source_extraction'));
addpath(genpath('~/Repositories/NoRMCorre'));
addpath(genpath('~/Repositories/Acquisition2P_class'));
addpath(genpath('~/Repositories/helperFunctions'));
addpath(genpath('~/Repositories/12k2p-software'));

if ~exist('~/Documents/MATLAB/cvx', 'dir')
    cd ~/MATLAB/cvx; cvx_setup;
else
    cd ~/Documents/MATLAB/cvx; cvx_setup;
end
cd ~/Repositories/2p-tester-scripts/matlab


