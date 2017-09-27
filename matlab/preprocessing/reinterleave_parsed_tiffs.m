function reinterleave_parsed_tiffs(mcparams)

split_channels = mcparams.split_channels;
switch mcparams.method
    case 'Acquisition2P'
        tmp_acq_obj = load(mcparams.info.acq_object_path);
        acq_obj = tmp_acq_obj.(mcparams.info.acquisition_name);
        clear tmp_acq_obj
    case 'NoRMCorre'
        % do some other stuff
    otherwise
        % maybe yet other stuff
end

% TODO: reinterleave_tiffs should support other MC methods. For now, just Acq2P, which deinterleaves TIFFs within itself.

% Re-interleave TIFF slices:
% split_channels = false;
reinterleave_tiffs(acq_obj, split_channels); %, info);
fprintf('Done reinterleaving TIFFs.\n')

end
