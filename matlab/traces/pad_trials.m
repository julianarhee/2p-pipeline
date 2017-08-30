function orig_cell = pad_trials(orig_cell, padded_mats, padded_idxs)
    orig_cell(padded_idxs) = padded_mats(1:end);
end
