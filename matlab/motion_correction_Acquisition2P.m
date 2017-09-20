gcp;

% run_multi_acquisitions=0;
% 
% crossref = false %true;
% processed = true; %true; %false % truea
% 
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/gratings1';
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/20161221_JR030W/test_crossref';
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/20161222_JR030W/gratings2/DATA';
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/test_motion_correction';
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/test_motion_correction_3D/DATA';
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/20170724_CE051/retinotopy1/DATA';
% %acquisition_dir = '/nas/volume1/2photon/RESDATA/20170811_CE052/retinotopy5';
% acquisition_dir = '/nas/volume1/2photon/RESDATA/20170825_CE055/fxnal_data/gratings1';
% 
% mc_ref_channel = 1; %1; %2;
% mc_ref_movie = 1; %1; %2;
% 


%if run_multi_acquisitions == 1
% acquisition_dirs = dir(acquisition_dir);
% isub = [acquisition_dirs(:).isdir]; %# returns logical vector
% acquisitions= {acquisition_dirs(isub).name}';
%else
%tiffs = dir(fullfile(acquisition_dir, '*.tif'));
%tiffs = {tiffs(:).name}';
%end

%tiffs(ismember(tiffs,{'.','..'})) = [];

%fprintf('Correcting %i movies: \n', length(tiffs));
%display(tiffs);

%for tiffidx=1:length(tiffs)
    
    % ---------------------------------------------------------------------
    % 1. Move each "acquisition" to be processed for M.C. into its own
    % directory:
%    currTiffName = tiffs{tiffidx};

%    currTiff = fullfile(acquisition_dir, currTiffName); 
    
%     if run_multi_acquisitions == 1
%         for tiff_idx = 1:length(currTiff)
%         curr_tiff_fn = currTiff{tiff_idx};
%         [pathstr,name,ext] = fileparts(curr_tiff_fn);
%         if ~exist(fullfile(curr_acquisition_dir, name), 'dir')
%             mkdir(fullfile(curr_acquisition_dir, name));
%             movefile(fullfile(curr_acquisition_dir, curr_tiff_fn), fullfile(curr_acquisition_dir, name, curr_tiff_fn));
%         end
%         end
%     end
    
fprintf('Processing acquisition %s...\n', acquisition_dir);
    % ---------------------------------------------------------------------
    % Walk through each acquisition-directory and run motion correction:
%     tiff_dirs = dir(curr_acquisition_dir);
%     tmp_isub = [tiff_dirs(:).isdir]; %# returns logical vector
%     tiffs = {tiff_dirs(tmp_isub).name}';
%     tiffs(ismember(tiffs,{'.','..'})) = [];
%     
%     for tiff_idx = 1:length(tiffs)
%     curr_mov = fullfile(curr_acquisition_dir, tiffs{tiff_idx});

if crossref
    myObj = Acquisition2P([],{@SC2Pinit_noUI_crossref,[],acquisition_dir,crossref});
    myObj.motionRefChannel = mc_ref_channel; %2;
    myObj.motionRefMovNum = mc_ref_movie;
    myObj.motionCorrectCrossref;
    %end
    myObj.save;
elseif processed
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],acquisition_dir});
    myObj.motionCorrectionFunction = @lucasKanade_plus_nonrigid; %withinFile_withinFrame_lucasKanade; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mc_ref_channel; %2;
    myObj.motionRefMovNum = mc_ref_movie;
    myObj.motionCorrectProcessed;
    %end
    myObj.save;
else
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],acquisition_dir});
    myObj.motionCorrectionFunction = @lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mc_ref_channel; %2;
    myObj.motionRefMovNum = mc_ref_movie;
    myObj.motionCorrect;
    %end
    myObj.save;
end

% Re-interleave TIFF slices:
split_channels = false;
interleaveTiffs(myObj, split_channels); %, info);

    
% ---------------------------------------------------------------------
% If using (and including) 2 channels for MC, separate them into their
% own dirs:
% if mc_ref_channel == 2
% %     for tiff_idx = 1:length(tiffs)
%     corrected_path = fullfile(acquisition_dir, 'Corrected');
%     corrected_tiff_fns = dir(fullfile(corrected_path, '*.tif'));
%     corrected_tiff_fns = {corrected_tiff_fns(:).name};
%     corrected_ch1_path = fullfile(corrected_path, 'Channel01');
%     corrected_ch2_path = fullfile(corrected_path, 'Channel02');
%     if ~exist(corrected_ch1_path, 'dir')
%         mkdir(corrected_ch1_path);
%         mkdir(corrected_ch2_path);
%     end
%     for tiff_idx=1:length(corrected_tiff_fns)
%         if strfind(corrected_tiff_fns{tiff_idx}, 'Channel01')
%             movefile(fullfile(corrected_path, corrected_tiff_fns{tiff_idx}), fullfile(corrected_ch1_path, corrected_tiff_fns{tiff_idx}));
%         else
%             movefile(fullfile(corrected_path, corrected_tiff_fns{tiff_idx}), fullfile(corrected_ch2_path, corrected_tiff_fns{tiff_idx}));
%         end
%     end
% %     end
% end
% 






