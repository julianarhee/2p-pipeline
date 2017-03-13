function save_struct(traces_path, traces_fn, T)

save(fullfile(traces_path, traces_fn), '-struct', 'T', '-v7.3');
fprintf('Saved struct %s to %s.\n', traces_fn, traces_path);

end