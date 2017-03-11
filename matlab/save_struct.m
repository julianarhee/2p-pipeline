function save_struct(traces_path, traces_fn, T)

save(fullfile(traces_path, traces_fn), 'T', '-struct', '-v7.3');
fprint('Saved struct %s to %s.\n', traces_fn, traces_path);

end