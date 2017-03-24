function save_struct(traces_path, traces_fn, T, varargin)

if isempty(varargin)
    save(fullfile(traces_path, traces_fn), '-struct', 'T', '-v7.3');
else
    save(fullfile(traces_path, traces_fn), '-append', '-struct', 'T');
end
fprintf('Saved struct %s to %s.\n', traces_fn, traces_path);

end