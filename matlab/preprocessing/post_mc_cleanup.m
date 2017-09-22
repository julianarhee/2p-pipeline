function post_mc_cleanup(mcparams, varargin)

% Sort Parsed files into separate directories if needed:
tmpchannels = dir(mcparams.corrected_dir);
tmpchannels = tmpchannels(arrayfun(@(x) ~strcmp(x.name(1),'.'), tmpchannels));
tmpchannels = tmpchannels([tmpchannels.isdir]);
tmpchannels = {tmpchannels(:).name}';
%if length(dir(fullfile(D.sourceDir, D.tiffSource, tmpchannels{1}))) > length(tmpchannels)+2
if isempty(tmpchannels) %|| any(strfind(D.tiffSource, 'Parsed'))
    sort_deinterleaved_tiffs(mcparams, varargin);
end
fprintf('Finished sorting parsed TIFFs.\n')        

end
