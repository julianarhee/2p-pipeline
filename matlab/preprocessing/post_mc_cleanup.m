function post_mc_cleanup(mcparams, varargin)

% Check whether 1st or 2nd MC step:
if length(varargin)==0
    bidi = false;
else
    bidi = varargin{1};
end

% Sort Parsed files into separate directories if needed:
tmpchannels = dir(mcparams.corrected_dir);
tmpchannels = tmpchannels(arrayfun(@(x) ~strcmp(x.name(1),'.'), tmpchannels));
tmpchannels = tmpchannels([tmpchannels.isdir]);
tmpchannels = {tmpchannels(:).name}';
%if length(dir(fullfile(D.sourceDir, D.tiffSource, tmpchannels{1}))) > length(tmpchannels)+2
if isempty(tmpchannels) %|| any(strfind(D.tiffSource, 'Parsed'))
    sort_deinterleaved_tiffs(mcparams, bidi);
end
fprintf('Finished sorting parsed TIFFs.\n')        

end
