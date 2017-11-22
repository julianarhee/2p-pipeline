% If already have ROIs or using auto-ROI method, can just do preprocessing & trace extraction with this script.

init_header;

initialize_analysis();

process_tiffs(I, A);

get_traces_from_rois(I, A);

fprintf('Completed both preprocessing and trace-extraction steps.\n')
fprintf('Summary of current analysis:\n')
display(I)
display(curr_mcparams)



