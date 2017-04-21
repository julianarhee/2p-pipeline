function origCell = padRoiTrials(origCell, paddedMats, paddedIdxs)
    origCell(paddedIdxs) = paddedMats(1:end);
end