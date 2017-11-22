function D = memmap3D(D, meta)

%TODO: Fix case when averaging runs together. Currently, memmapping for averaged runs is true.
 
if D.processedtiffs
    create_substack = false;
    startSliceIdx = 1;
else
    create_substack = true;
    startSliceIdx = D.slices(1);
    fprintf('Creating substacks made stareting from slice %i.\n', startSliceIdx);
end
% if create_substack
%     startSliceIdx = D.slices(1);
% else
%     startSliceIdx = 1;
% end
fprintf('Substacks made starting from slice %i.\n', startSliceIdx);

memmapped = D.memmapped;

% Get all TIFFs to be processed and stored as memmapped file:
% -------------------------------------------------------------------------
tiffs = dir(fullfile(D.dataDir, '*.tif'));
fprintf('Getting TIFFs from alternate data dir: %s', D.dataDir);
tiffDir = D.dataDir;
tiffs = {tiffs(:).name}'

 
% -------------------------------------------------------------------------
% STEP 1:  TIFFS to memmapped .mat files:
% -------------------------------------------------------------------------

if memmapped
    extension = 'mat'
else
    extension = 'tif';
end

% Check if D.mempath already contains memmaped (.mat) files:
% -------------------------------------------------------------------------
tmpfiles = dir(fullfile(D.mempath, strcat('*.', extension)));
tmpfiles = {tmpfiles(:).name}';
if create_substack
    subidxs = cell2mat(cellfun(@(x) any(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0))
    files = tmpfiles(subidxs)
else
    files = tmpfiles;
end

if memmapped
    fprintf('memmapping...\n')
    % If no memmaped files found in D.mempath, make them for each TIFF:
    % -------------------------------------------------------------------------
    if isempty(files) || isempty(tmpfiles) || length(tiffs)>length(files) % in case NOT using '_substack):
        
        % NO memmat files found, create new from TIFFs:
        tic()
        for tiffidx = 1:length(tiffs)
            tpath = fullfile(tiffDir, tiffs{tiffidx});
            fprintf('Processing tiff %i of %i to mem from path:\n%s\n', tiffidx, length(tiffs), tpath);
            [source, filename, ext] = fileparts(tpath);

            matpath = fullfile(D.mempath, sprintf('%s.mat', filename));
            data = matfile(matpath,'Writable',true);
            
            nSlices = meta.file(tiffidx).si.nFramesPerVolume
            nRealFrames = meta.file(tiffidx).si.nSlices
            nVolumes = meta.file(tiffidx).si.nVolumes
            nChannels = meta.file(tiffidx).si.nChannels
         
             
            if D.metaonly && ~D.processedtiffs % i.e., tiff is too huge to load into matlab
                fprintf('File too large, reading in single slice at at ime...\n');
     
                % Since too large (for Matlab), already parsed in Fiji:
                %tiffsourcePath = fullfile(D.sourceDir, D.tiffSource, 'Channel01', sprintf('File%03d', tiffidx));
                tiffsourcePath = fullfile(D.tiffSource, 'Channel01', sprintf('File%03d', tiffidx));
                tiffslices = dir(fullfile(tiffsourcePath, '*.tif'));
                tiffslices = {tiffslices(:).name}';
                
                Yt = loadtiff(fullfile(tiffsourcePath, tiffslices{1}));
                
                data.Y(size(Yt,1), size(Yt,2), nRealFrames, nVolumes) = Yt(1)*0; 
                data.Yr(size(Yt,1)*size(Yt,2)*nRealFrames, nVolumes) = Yt(1)*0;
                
                for sliceidx=1:length(tiffslices)
                    tic(); tmpYt = loadtiff(fullfile(tiffsourcePath, tiffslices{sliceidx})); toc();
                    data.Y(:,:,sliceidx,:) = reshape(tmpYt, [size(Yt,1) size(Yt,2) 1 nVolumes]);
                end
                if ~isa(data.Y, 'double'); data.Y = double(data.Y); end % convert to double
                data.Y = data.Y - min(data.Y(:));                       % make data non-negative
                if D.correctbidi
                    data.Y = correct_bidirectional_phasing(data.Y);
                end
                
                sizY = size(data.Y);
                data.Yr(1:prod(sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(sizY(1:end-1)),[]);
                 
                data.sizY = sizY;	
                data.nY = min(min(data.Yr(:,:)));
                fprintf('size nY: %s\n.', mat2str(size(data.nY)))
                %data.Yr(1:prod(data.sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(data.sizY(:,1:end-1)),[]);
                %data.nY = min(data.Yr(:,:));
                
                fprintf('Memmapping finished for %i of %i files.\n', tiffidx, length(tiffs));
                fprintf('Size of memmaped movie is: %s\n', mat2str(data.sizY));

            else % ok to read in whole thing
                
                % tic; Yt = loadtiff(tpath, 1, nSlices*nVolumes*nChannels); toc;
                tic; Yt = read_file(tpath); toc; % is this faster
     
                % Only grab green channel:
                if strcmp(D.preprocessing, 'raw') || D.processedtiffs %&& D.tefo
                    fprintf('Grabbing every other channel.\n')
                    Yt = Yt(:,:,1:D.nChannels:end);
                    fprintf('Single channel, mov size is: %s', mat2str(size(Yt)));
                end

                Y = cell(1, nVolumes);
                firstslice = startSliceIdx; %1;
                for vol=1:nVolumes
                    Y{vol} = Yt(:,:,firstslice:(firstslice+nRealFrames-1));
                    firstslice = firstslice+nSlices;
                end
                Y = cat(4, Y{1:end});
                if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
                Y = Y -  min(Y(:));                         % make data non-negative
                if D.correctbidi
		    fprintf('Correcting bidirectional scanning offset.\n');
                    Y = correct_bidirectional_phasing(Y);
                end
                tiffWrite(Y, strcat(filename, '.tif'), D.mempath) 

                data.Y = Y;
                sizY = size(data.Y);
                data.Yr(1:prod(sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(sizY(1:end-1)),[]);
                data.sizY = sizY;
                data.nY = min(min(data.Yr(:,:)));
                %data.Yr(1:prod(data.sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(data.sizY(:,1:end-1)),[]);
                %data.nY = min(data.Yr(:,:));
                fprintf('size nY: %s\n.', mat2str(size(data.nY)))

                fprintf('Memmapping finished for %i of %i files.\n', tiffidx, length(tiffs));
                fprintf('Size of memmaped movie is: %s\n', mat2str(data.sizY));
            end            
        end
    end
else
    for tiffidx = 1:length(tiffs)
        fprintf('Not creating memmapped files.\n');
        tpath = fullfile(tiffDir, tiffs{tiffidx});
        fprintf('Processing tiff from path:\n%s\n', tpath);
        [source, filename, ext] = fileparts(tpath);

        % tic; Yt = loadtiff(tpath, 1, nSlices*nVolumes*nChannels); toc;
        tic; Yt = read_file(tpath); toc; % is this faster

        % Only grab green channel:
        if strcmp(D.preprocessing, 'raw') || D.processedtiffs %&& D.tefo
            fprintf('Grabbing every other channel.\n')
            Yt = Yt(:,:,1:D.nChannels:end);
            fprintf('Single channel, mov size is: %s\n', mat2str(size(Yt)));
        end

        Y = cell(1, nVolumes);
        firstslice = startSliceIdx; %1;
        for vol=1:nVolumes
            Y{vol} = Yt(:,:,firstslice:(firstslice+nRealFrames-1));
            firstslice = firstslice+nSlices;
        end
        Y = cat(4, Y{1:end});
        if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
        Y = Y -  min(Y(:));                         % make data non-negative
        if D.correctbidi
	    fprintf('Correcting bidirectional scanning offset.\n');
            Y = correct_bidirectional_phasing(Y);
        end
        tiffWrite(Y, strcat(filename, '.tif'), D.mempath)
    end
end

% -------------------------------------------------------------------------
% STEP 2:  AVERAGE, if needed.
% -------------------------------------------------------------------------
% Now, get memmaped file names again, and check for averaging conditions
% (i.e., Step 1 creates .mat for every TIFF, regardless).  Here, create
% memapped files for "averaged" tiffs using the original .mat files:
% -------------------------------------------------------------------------

if D.average
    checkfiles = dir(fullfile(D.averagePath, strcat('*.', extension)));
    checkfiles = {checkfiles(:).name}';
    if memmapped
        inputfiles = dir(fullfile(D.mempath, '*.mat'));
    else
        inputfiles = dir(fullfile(D.mempath, '*.tif'));
    end
    inputfiles = {inputfiles(:).name}';

    if isempty(checkfiles) % AVERAGE:

        if create_substack
            subidxs = cell2mat(cellfun(@(x) any(strfind(x, '_substack')), inputfiles, 'UniformOutput', 0));
            files4averaging = inputfiles(subidxs);
        else
            files4averaging = inputfiles;
        end

        % For now, there is 1 acquisition ("run" in DB lingo) that we are
        % counting as the primary run.  Use this to appropriate match and index
        % the files from whatever other run(s) we are averaging.
        fileidxs1 = cell2mat(cellfun(@(f) any(strfind(f, D.acquisitionName)), files4averaging, 'UniformOutput', 0));
        files1 = files4averaging(fileidxs1);
        files2 = files4averaging(~fileidxs1);
        
        matchtrials = D.matchtrials;
        fprintf('Averagin tiffs...\n')
        for runidx=1:length(matchtrials)
            curr_match_idxs = matchtrials{runidx};
            if memmapped
                filename = sprintf('File%03d_averagerun.mat', runidx);
                filedata = matfile(fullfile(D.averagePath, filename), 'Writable', true);
                f1 = matfile(fullfile(D.mempath, files1{curr_match_idxs(1)}));
                f2 = matfile(fullfile(D.mempath, files2{curr_match_idxs(2)}));
                
                newfiletmp = cat(5, f1.Y, f2.Y);
                avgfile = mean(newfiletmp, 5);
                [d1,d2,d3,t] = size(avgfile)
                filedata.Y(d1,d2,d3,t) = avgfile(1)*0;
                filedata.Y(:,:,:,:) = avgfile;
                filedata.Yr(d1*d2*d3, t) = avgfile(1)*0;
                filedata.Yr(:,:) = reshape(avgfile, d1*d2*d3,[]);
                filedata.nY = min(filedata.Yr(:,:));
                clear newfiletmp avgfile
            else
                filename = sprintf('File%03d_averagerun.tif', runidx);
                f1 = read_file(fullfile(D.dataDir, files1{curr_match_idxs(1)}));
                f2 = read_file(fullfile(D.dataDir, files2{curr_matchidxs(2)}));
                newfiletmp = cat(5, f1, f2);
                avgfile = mean(newfiletmp, 5);
                tiffWrite(avgfile, filename, D.averagePath);
                
            end
            fprintf('Primary file is: %s\n', files1{curr_match_idxs(1)});
            fprintf('Secondary file is: %s\n', files2{curr_match_idxs(2)});

        end

    end
    
end
 
fprintf('Done!');


% -------------------------------------------------------------------------
% STEP 3:  Create SUBSTACK, if needed.
% -------------------------------------------------------------------------
% Now, get memmaped file names again, and check for averaging conditions
% (i.e., Step 1 creates .mat for every TIFF, regardless).  Here, create
% memapped files for "averaged" tiffs using the original .mat files:
% -------------------------------------------------------------------------

if D.average
    mempath = D.averagePath;
else
    mempath = D.mempath; %fullfile(D.nmfPath, 'memfiles');
end

tmpfiles = dir(fullfile(mempath, '*.mat'));
tmpfiles = {tmpfiles(:).name}';
% % if create_substacks
% subidxs = cell2mat(cellfun(@(x) ~isempty(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0))
% inputfiles = tmpfiles(subidxs);
% % else
% %     inputfiles = tmpfiles;

if create_substack
    subidxs = cell2mat(cellfun(@(x) any(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0));
    inputfiles = tmpfiles(subidxs);
else
    inputfiles = tmpfiles;
    subidxs = [1:length(inputfiles)]
end


if isempty(inputfiles)
    files = tmpfiles(~subidxs);

    for tiffidx = 1:length(files)

        fprintf('Processing substack for %i of %i files...\n', tiffidx, length(files))

        first_tic = tic();

        nSlices = meta.file(tiffidx).si.nFramesPerVolume;
        nRealFrames = meta.file(tiffidx).si.nSlices;
        nVolumes = meta.file(tiffidx).si.nVolumes;
        if D.tefo
            nChannels=2;
        else
            nChannels=1;
        end

        tpath = fullfile(mempath, files{tiffidx});
        [filepath, filename, ext] = fileparts(tpath);

        tmpdata = matfile(tpath,'Writable',true);

        filename = [filename, '_substack']
        data = matfile(fullfile(filepath, [filename, ext]), 'Writable', true);
        % CROP flybacK;
        data.Y = tmpdata.Y(:,:,startSliceIdx:end,:);
        data.sizY = size(data.Y);
        data.Yr(1:prod(data.sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(data.sizY(:,1:end-1)),[]);
        data.nY = min(data.Yr(:,:));

        clear tmpdata;

        fprintf('Finished:  substack size is %s\n', mat2str(data.sizY));

    end
end


if create_substack
    subidxs = cell2mat(cellfun(@(x) any(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0));
    inputfiles = tmpfiles(subidxs);
else
    inputfiles = tmpfiles;
end

fprintf('Checking TIFF substack size...\n');

for tiffidx=1:length(inputfiles)
    tpath = fullfile(mempath, inputfiles{tiffidx});
    if memmapped
        data = matfile(tpath, 'Writable', true);
        if ~isprop(data, 'sizY')
            data.sizY = size(data.Y);
        end
        if length(data.nY)>1
            data.nY = min(data.nY(:,:));
        end
        if isprop(data, 'F_dark') && length(data.F_dark)>1
            data.F_dark = min(data.F_dark(:,:));
        end
        fprintf('TIFF %i of %i: size is %s.\n', tiffidx, length(inputfiles), mat2str(data.sizY));
        d1=data.sizY(1,1); d2=data.sizY(1,2); d3=data.sizY(1,3);
    else
        tiffdata = read_file(tpath);
        d1=size(tiffdata,1); d2=size(tiffdata,2); d3=size(tiffdata,3);
        fprintf('TIFF %i of %i: size is %s.\n', tiffidx, length(inputfiles), mat2str(size(tiffdata)));
    end
    [path_to_averages, ref_file, ~] = fileparts(D.sliceimagepath);
    pathbyfile = fullfile(path_to_averages, sprintf('File%03d', tiffidx));
    if ~exist(pathbyfile) || length(dir(fullfile(pathbyfile, '*.tif')))<data.sizY(1,3)
        if ~exist(pathbyfile, 'dir')
            mkdir(pathbyfile)
        end 
        fprintf('Averaging slices...\n');
        
        avgs = zeros([d1,d2,d3]);
        for slice=1:d3
            if memmapped
                avgs(:,:,slice) = mean(data.Y(:,:,slice,:), 4);
            else
                avgs(:,:,slice) = mean(tiffdata(:,:,slice,:), 4);
            end
            slicename = sprintf('average_Slice%02d_File%03d.tif', slice, tiffidx);
            tiffWrite(avgs(:,:,slice), slicename, pathbyfile);
        end
    end
end

    
end

