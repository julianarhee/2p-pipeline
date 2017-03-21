function [interpTraces] = interpolateTraces(dtimes, dtraces, reftimes)


        %roiTraces = rawTracemat{1};
        %for trial=1:length(dtraces)
        F = griddedInterpolant(dtimes, dtraces);
            %F = griddedInterpolant(siTimes.(currStim){trial}, roiTraces{trial});


            %nRois = size(siTraces.(currStim){trial},1);
        interpTraces = F(reftimes);
        %end
        
        %interpTraceMat = cat(1, iTraces.(currStim){1:end}); % concatenates each trial's ROI traces along rows.

end