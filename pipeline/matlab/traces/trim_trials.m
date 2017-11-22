function orig_cell = trim_trials(orig_cell, replacement_mats, replacement_idxs)
    orig_cell(replacement_idxs) = replacement_mats(1:end);
end
