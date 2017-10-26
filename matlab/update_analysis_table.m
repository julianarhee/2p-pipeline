function updatedI = update_analysis_table(itable, path_to_record)



curr_vars = itable.Properties.VariableNames;
 
% turn mats into strings:
for v=1:length(curr_vars)
    if any(size(itable.(curr_vars{v}))>1) && ~ischar(itable.(curr_vars{v})) && isnumeric(itable.(curr_vars{v}))
        itable.(curr_vars{v}) = mat2str(itable.(curr_vars{v}));
    end
%    if iscell(itable.(curr_vars{v}))
%        itable.(curr_vars{v}) = itable.(curr_vars{v}){1};
%    end
end

if exist(path_to_record, 'file')
    existing = readtable(path_to_record, 'Delimiter', '\t', 'ReadRowNames', true);
    existing_vars = existing.Properties.VariableNames;
% 
%     for v=1:length(existing_vars)
%         if any(size(existing.(existing_vars{v}))>1) && ~ischar(existing.(existing_vars{v})) && isnumeric(existing.(existing_vars{v}))
%             existing.(existing_vars{v})
%             existing.(existing_vars{v}) = mat2str(existing.(existing_vars{v}));
%         end
% %        if iscell(existing.(existing_vars{v}))
% %            existing.(existing_vars{v}) = existing.(existing_vars{v}){1};
% %        end
%     end


%     for v=1:length(existing_vars)
%         if any(size(existing.(existing_vars{v}))>1) && ~iscell(existing.(existing_vars{v}))
%             existing_vars{v}
%             existing.(existing_vars{v}) = mat2str(existing.(existing_vars{v}));
%         end
%     end
% 

    if length(curr_vars) > length(existing_vars)
         new_vars = find(arrayfun(@(i) ~ismember(curr_vars{i}, existing_vars), 1:length(curr_vars)));
        % append empty column of missing vars to existing table:
        [nruns, nvars] = size(existing);
        T = struct();
        for new = new_vars
            new_varname = curr_vars{new};
            new_col = ''; %zeros(nruns, 1);
            T.(new_varname) = new_col;
        end
        columns_to_add = struct2table(T);
        existing = [existing columns_to_add];
    end

    prev_ids = existing.Properties.RowNames;
    curr_id = itable.Properties.RowNames;
    if any(ismember(prev_ids, curr_id))
        row_idx = find(arrayfun(@(i) strcmp(prev_ids{i}, curr_id), 1:length(prev_ids)))
        existing(row_idx, :) = itable; 
        %existing(itable.Properties.RowNames,:) = itable; 
        updatedI = existing;
    else
        updatedI = [existing; itable];
    end
else
    updatedI = itable;
end

writetable(updatedI, path_to_record, 'Delimiter', '\t', 'WriteRowNames', true); 
