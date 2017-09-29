function post_mc_cleanup(path_to_sort, A)

% % Check whether 1st or 2nd MC step:
% if length(varargin)==0
%     bidi = false;
% else
%     bidi = varargin{1};
% end
% 
% Sort Parsed files into separate directories if needed:
% if bidi
%     tmpchannels = dir(fullfile(mcparams.tiff_dir, mcparams.bidi_corrected_dir));
% else
%     tmpchannels = dir(fullfile(mcparams.tiff_dir, mcparams.corrected_dir));
% end

tmpchannels = dir(fullfile(path_to_sort, '*.tif'));
tmpchannels = tmpchannels(arrayfun(@(x) ~strcmp(x.name(1),'.'), tmpchannels));
tmpchannels = tmpchannels([tmpchannels.isdir]);
tmpchannels = {tmpchannels(:).name}';

if isempty(tmpchannels) %|| any(strfind(D.tiffSource, 'Parsed'))
    sort_deinterleaved_tiffs(path_to_sort, A);
end
fprintf('Finished sorting parsed TIFFs.\n')        

end
