function updatedI = update_analysis_table(existsI, itable, path_to_fn)

curr_vars = itable.Properties.VariableNames;
% turn mats into strings:
for v=1:length(curr_vars)
    if any(size(itable.(curr_vars{v}))>1)
        itable.(curr_vars{v}) = mat2str(itable.(curr_vars{v}));
    end
end

if isempty(existsI)
    updatedI = itable;

else
    existing_vars = existsI.Properties.VariableNames;
    
    for v=1:length(existing_vars)
        if any(size(existsI.(existing_vars{v}))>1)
            existsI.(existing_vars{v}) = mat2str(existsI.(existing_vars{v}));
        end
    end


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
        existsI = [existsI columns_to_add];
    end


    if any(ismember(existsI.Properties.RowNames, itable.Properties.RowNames))
        updatedI = existsI;
    else
        updatedI = [existsI; itable];
    end
end

writetable(updatedI, path_to_fn, 'Delimiter', '\t', 'WriteRowNames', true);

