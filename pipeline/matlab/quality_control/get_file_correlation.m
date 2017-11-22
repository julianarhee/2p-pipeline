function [mean_file_corr] = get_file_correlation(tracestruct)
%takes in:
%   tracestruct - structure with raw timecourse for each roi

%returns
%   mean_file_corr - vector with file-to-file time correlations averaged
%   across files for each ROI

nFiles = length(tracestruct.file);
%initialize emptry matrix
mean_file_corr = zeros(1,tracestruct.file(1).nrois);

%loop through ROIs
for roi = 1:tracestruct.file(1).nrois
    %extract tiemcourse for all files
    roi_trace = zeros(nFiles, size(tracestruct.file(1).rawtracemat,1));
    for fidx = 1:nFiles
        roi_trace(fidx,:) = tracestruct.file(fidx).rawtracemat(:,roi);
    end
    
    %get file-to-file correlation matrix
    R= corr(roi_trace');

    %get mean file-to-file correlation 
    R_ind = find(triu(ones(size(R)),1)==1);
    mean_file_corr(roi)=mean(R(R_ind));
end

