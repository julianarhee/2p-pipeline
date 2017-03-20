function thresholdMap = threshold_map(image, filter, threshold)

    thresholdMap = image;
    thresholdMap(filter<(max(filter(:))*threshold)) = NaN;

end