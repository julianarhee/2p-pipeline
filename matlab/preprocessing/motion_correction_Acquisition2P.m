function myObj = motion_correction_Acquisition2P(mcparams)
%gcp;

% MC params are set in set_mc_params.m
% -------------------------------------------------------------------------
% mcparams.acqusition_dir - directory containing all the pre-MC TIFFs (this
% will either be the Acqusition dir itself, or 'DATA/' if substacks created
% with create_substacks.py

% mcparams.crossref - coregister across acquisitions that are the same FOV
% [true or false]
% 
% mcparams.processed - using raw tiffs or substacks made from
% create_substacks.py [true or false]
% 
% mcparams.ref_channel - if multi-channel, which channel to use as ref
% [default: 1]
% 
% mcparams.ref_movie  - if multiple TIFFs in current acqusition, index
% (File00x) of movie to use as reference [default: 1]
% 
% mcparams.algorithm - which correction algo to use. Options:
% Acq2P:  @lucasKanade_plus_nonrigid | @withinFile_withinFrame_lucasKanade
% NoRMCorre:  rigid | nonrigid

fprintf('Processing acquisition %s...\n', mcparams.acquisition_dir);

if mcparams.crossref
    myObj = Acquisition2P([],{@SC2Pinit_noUI_crossref,[],mcparams.acquisition_dir,mcparams.crossref});
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_movie;
    myObj.motionCorrectCrossref;
    myObj.save;
elseif mcparams.processed
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],mcparams.acquisition_dir});
    myObj.motionCorrectionFunction = mcparams.algorithm; %@lucasKanade_plus_nonrigid; %withinFile_withinFrame_lucasKanade; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_movie;
    myObj.motionCorrectProcessed;
    myObj.save;
else
    myObj = Acquisition2P([],{@SC2Pinit_noUI,[],mcparams.acquisition_dir});
    myObj.motionCorrectionFunction = mcparams.algorithm; %@lucasKanade_plus_nonrigid;
    myObj.motionRefChannel = mcparams.ref_channel; %2;
    myObj.motionRefMovNum = mcparams.ref_movie;
    myObj.motionCorrect;
    myObj.save;
end
    
%end






