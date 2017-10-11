function updatedI = update_analysis_table(existsI, itable, path_to_fn)


existing_vars = existsI.Properties.VariableNames;
curr_vars = itable.Properties.VariableNames;
if length(curr_vars) > length(existing_vars)
    % append empty column of missing vars to existing table:
    [nruns, nvars] = size(existsI);
    new_vars = find(arrayfun(@(i) ~ismember(curr_vars{i}, existing_vars), 1:length(curr_vars)));
    T = struct();
    for new = new_vars
        new_varname = curr_vars{new};
        new_col = zeros(nruns, 1);
        T.(new_varname) = new_col;
    end
    columns_to_add = struct2table(T);
    existsI = [existsI columns_to_add]
end
        
updatedI = [existsI; itable];
writetable(updatedI, path_to_fn, 'Delimiter', '\t', 'WriteRowNames', true);
