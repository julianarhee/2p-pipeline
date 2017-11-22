function [trace_y, DC_y] = subtract_rolling_mean(curr_trace, winsz)
    % if cutEnd==1
    %     tmp0 = curr_trace(1:crop);
    % else
    ind = 1:length(curr_trace);
    ix = ~isnan(curr_trace);
    if sum(ix) ~= length(curr_trace)
        tmp0 = interp1(ind(ix),curr_trace(ix),ind,'linear').';
        % transpose so that in output, cols are still rois and rows are
        % still frames
    else
        tmp0 = curr_trace;
    end
    %tmp0 = curr_trace;
    % end
    if size(tmp0,1) > size(tmp0,2)
        tmp1 = padarray(tmp0,[winsz 0],tmp0(1),'pre');
        tmp1 = padarray(tmp1,[winsz 0],tmp1(end),'post');
        rolling_avg=nanconv(tmp1,fspecial('average',[winsz 1]),'same');%average
    else
        tmp1 = padarray(tmp0,[0 winsz],tmp0(1),'pre');
        tmp1 = padarray(tmp1,[0 winsz],tmp1(end),'post');
        rolling_avg=nanconv(tmp1,fspecial('average',[1 winsz]),'same');%average
    end
    rolling_avg=rolling_avg(winsz+1:end-winsz);
    trace_y = tmp0 - rolling_avg;
    
    DC_y = mean(tmp0);
    
end
