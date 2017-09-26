function info = do_motion_correction(mcparams)

% TODO: set case statements to choose either Acquisition2P or NoRMCorre here.
info = struct();

switch mcparams.method
    
    case 'Acquisition2P'
        gcp;
        acqObj = motion_correction_Acquisition2P(mcparams);
        %[corrected_path, ~, ~] = fileparts(acqObj.correctedMovies.slice(1).channel(1).fileName{1});
        %mcparams.corrected_dir = corrected_path;
        info.acquisition_name = acqObj.acqName;
        info.acq_object_path = fullfile(acqObj.defaultDir, strcat('Acq_', acqObj.acqName));
        
    case 'NoRMCorre'
        
        % do stuff
    
    otherwise
        fprintf('No motion-correction source specified.\n');
        
end

end