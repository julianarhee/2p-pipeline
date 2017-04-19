function origCell = trimRoiTrials(origCell, replacementMats, replacementIdxs)
    origCell(replacementIdxs) = replacementMats(1:end);
end