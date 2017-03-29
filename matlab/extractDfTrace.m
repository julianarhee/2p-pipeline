function [currDf] = extractDfTrace(currTrace)

method = 'simple';
    switch method
        case 'simple'
            currDf = (currTrace - mean(currTrace)) ./ mean(currTrace);
        case 'simple_baseline'
            % do stuff
    end

end