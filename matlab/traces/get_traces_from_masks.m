function tracemat = get_traces_from_masks(Y, maskcell)
    
    maskfunc = @(x,y) sum(x(y)); % way faster
    cellY = num2cell(Y, [1 2]);
    
    % For each frame of the movie, apply each ROI mask:
    tracemat_tmp = squeeze(cellfun(@(frame) cellfun(@(c) maskfunc(frame, c), maskcell), cellY, 'UniformOutput', false));
    tracemat = cat(1, tracemat{1:end});
    
end