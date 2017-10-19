function myObj = motion_correction_Acquisition2P(mcparams)
%gcp;

% MC params are set in set_mc_params.m
% -------------------------------------------------------------------------
% mcparams.tiff_dir - directory containing all the pre-MC TIFFs (this
% will 'DATA/', which should be a child of the Acqusition dir.

% mcparams.crossref - coregister across acquisitions that are the same FOV
% [true or false]
% 
% mcparams.flyback_corrected - using raw tiffs or substacks made from
% create_substacks.py [true or false]
% 
% mcparams.ref_channel - if multi-channel, which channel to use as ref
% [default: 1]
% 
% mcparams.ref_file  - if multiple TIFFs in current acqusition, index
% (File00x) of movie to use as reference [default: 1]
% 
% mcparams.algorithm - which correction algo to use. Options:
% Acq2P:  @lucasKanade_plus_nonrigid | @withinFile_withinFrame_lucasKanade
% NoRMCorre:  rigid | nonrigid

%fprintf('Processing acquisition %s...\n', mcparams.source_dir);

check_tiffs = dir(fullfile(mcparams.source_dir, '*.tif'));
if length(check_tiffs)==0
    tiff_dir = fullfile(mcparams.source_dir, 'Raw');
    check_tiffs = dir(fullfile(tiff_dir, '*.tif'));
else
    tiff_dir = mcparams.source_dir;
end
write_dir = fullfile(mcparams.source_dir, mcparams.dest_dir);

fprintf('Processing acquisition %s...\n', tiff_dir);
fprintf('Found %i TIFFs.\n', length(check_tiffs));
  
if mcparams.crossref
    myObj = Acquisition2P([],{@SC2Pinit_noUI_crossref,[],tiff_dir,mcparams.crossref});
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_file;
    myObj.motionCorrectCrossref;
    myObj.save;
elseif mcparams.flyback_corrected
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],tiff_dir});
    myObj.motionCorrectionFunction = mcparams.algorithm; %@lucasKanade_plus_nonrigid; %withinFile_withinFrame_lucasKanade; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_file;
    myObj.motionCorrectProcessed(write_dir);
    myObj.save;
else
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],tiff_dir});
    myObj.motionCorrectionFunction = mcparams.algorithm; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_file;
    %myObj.motionCorrect;
    myObj.motionCorrectProcessed(write_dir);
    myObj.save;
end
    
%end






