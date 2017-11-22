function [currDf] = extract_df(curr_trace)

method = 'simple';
    switch method
        case 'simple'
            currDf = (curr_trace - mean(curr_trace)) ./ mean(curr_trace);
        case 'simple_baseline'
            % do stuff
    end

end
