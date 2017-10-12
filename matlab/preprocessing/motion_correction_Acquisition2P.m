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

fprintf('Processing acquisition %s...\n', mcparams.tiff_dir);

if mcparams.crossref
    myObj = Acquisition2P([],{@SC2Pinit_noUI_crossref,[],mcparams.tiff_dir,mcparams.crossref});
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_file;
    myObj.motionCorrectCrossref;
    myObj.save;
elseif mcparams.flyback_corrected
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],mcparams.tiff_dir});
    myObj.motionCorrectionFunction = mcparams.algorithm; %@lucasKanade_plus_nonrigid; %withinFile_withinFrame_lucasKanade; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_file;
    myObj.motionCorrectProcessed;
    myObj.save;
else
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],mcparams.tiff_dir});
    myObj.motionCorrectionFunction = mcparams.algorithm; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_file;
    %myObj.motionCorrect;
    myObj.motionCorrectProcessed;
    myObj.save;
end
    
%end






