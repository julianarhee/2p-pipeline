%% Clear all and make sure paths set
clc; clear all;

% add repo paths
add_repo_paths
fprintf('Added repo paths.\n');

%% Create Acquisition Struct:

A.source = '/nas/volume1/2photon/projects/gratings_phaseMod';
A.session = '20170901_CE054_zoom3x';
A.acquisition = 'functional_test';

%% Preprocess data:

% do_preprocessing();

