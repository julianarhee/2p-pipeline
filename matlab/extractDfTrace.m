function [currDf] = extractDfTrace(currTrace)

	currDf = (currTrace - mean(currTrace)) ./ mean(currTrace);

end