function mcparams = do_motion_correction(mcparams)

% TODO: set case statements to choose either Acquisition2P or NoRMCorre here.

switch mcparams.source
    
    case 'Acquisition2P'
        acqObj = motion_correction_Acqusition2P(mcparams);
        [corrected_path, ~, ~] = fileparts(acqObj.correctedMovies.slice(1).channel(1).fileName{1});
        mcparams.output_dir = corrected_path;
        mcparams.acquisition_name = acqObj.acqName;
        mcparams.nchannels = length(acqObj.correctedMovies.slice(1).channel);
        mcparams.acq_object_path = fullfile(acqObj.defaultDir, strcat('Acq_', acqObj.acqName));
        
    case 'NoRMCorre'
        
        % do stuff
    
    otherwise
        fprintf('No motion-correction source specified.\n');
        
end

fprintf('Completed motion-correction!\n');

end