function reinterleave_parsed_tiffs(mcparams)

split_channels = mcparams.split_channels;
tmp_acq_obj = load(mcparams.acq_object_path);
acq_obj = tmp_acq_obj.(mcparams.acquisition_name);
clear tmp_acq_obj

% Re-interleave TIFF slices:
% split_channels = false;
reinterleave_tiffs(acq_obj, split_channels); %, info);
fprintf('Done reinterleaving TIFFs.\n')

end
