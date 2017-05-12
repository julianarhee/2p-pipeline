function memmap3D(D, meta)

startSliceIdx = D.slices(1);
fprintf('Substacks made starting from slice %i.\n', startSliceIdx);

mempath = fullfile(D.nmfPath, 'memfiles');
if ~exist(mempath, 'dir')
    mkdir(mempath)
end


tiffs = dir(fullfile(D.sourceDir, '*.tif'));
tiffs = {tiffs(:).name}'

files = dir(fullfile(mempath, '*.mat'));
tmpfiles = {files(:).name}';
%subidxs = cell2mat(cellfun(@(x) isempty(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0))
subidxs = cell2mat(cellfun(@(x) any(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0))
files = tmpfiles(subidxs)

if D.average
    % For now, there is 1 acquisition ("run" in DB lingo) that we are
    % counting as the primary run.  Use this to appropriate match and index
    % the files from whatever other run(s) we are averaging.
    fileidxs1 = cell2mat(cellfun(@(f) any(strfind(f, D.acquisitionName)), files, 'UniformOutput', 0));
    files1 = files(fileidxs1);
    files2 = files(~fileidxs1);
end
    
checkfiles = dir(fullfile(D.averagePath, '*.mat'));
checkfiles = {checkfiles(:).name}';

if length(tiffs)<length(files) && D.average && isempty(checkfiles) % AVERAGE:
    matchtrials = D.matchtrials;
    fprintf('Averagin tiffs...\n')
    for runidx=1:length(matchtrials)
        curr_match_idxs = matchtrials{runidx};
        filename = sprintf('File%03d_substack.mat', runidx);
        avgdir = D.averagePath; %fullfile(D.datastructPath, 'averaged');

        matfilepath = fullfile(avgdir, filename);
        filedata = matfile(matfilepath, 'Writable', true);
        f1 = matfile(fullfile(mempath, files1{curr_match_idxs(1)}));
        f2 = matfile(fullfile(mempath, files2{curr_match_idxs(2)}));
        fprintf('Primary file is: %s\n', files1{curr_match_idxs(1)});
        fprintf('Secondary file is: %s\n', files2{curr_match_idxs(2)});
        
        newfiletmp = cat(5, f1.Y, f2.Y);
        avgfile = mean(newfiletmp, 5);
        [d1,d2,d3,t] = size(avgfile)
        filedata.Y(d1,d2,d3,t) = avgfile(1)*0;
        filedata.Y(:,:,:,:) = avgfile;
        filedata.Yr(d1*d2*d3, t) = avgfile(1)*0;
        filedata.Yr(:,:) = reshape(avgfile, d1*d2*d3,[]);
        filedata.nY = min(filedata.Yr(:,:));
        clear newfiletmp avgfile
    end

elseif length(tiffs)>length(files)


    tic()
    for tiffidx = 1:length(tiffs)
    %tpath = '/nas/volume1/2photon/RESDATA/20161222_JR030W/gratings1/fov1_gratings_10reps_run1_00007.tif';
    tpath = fullfile(D.sourceDir, tiffs{tiffidx});
    [source, filename, ext] = fileparts(tpath);

    matpath = fullfile(mempath, sprintf('%s.mat', filename));
    data = matfile(matpath,'Writable',true);

    %info = imfinfo(tpath);

    %nSlices = 20; nVolumes = 350;
    %tic; Yt = loadtiff(tpath, 1, nSlices*2); toc;
    %Yt = Yt(:,:,1:2:end);

    %nSlices = meta.file(tiffidx).si.nSlices;
    nSlices = meta.file(tiffidx).si.nFramesPerVolume;
    nRealFrames = meta.file(tiffidx).si.nSlices;
    nVolumes = meta.file(tiffidx).si.nVolumes;
    if D.tefo
        nChannels=2;
    else
        nChannels=1;
    end
    
    if D.metaonly % i.e., tiff is too huge to load into matlab
        
        % Since too large (for Matlab), already parsed in Fiji:
        tiffsourcePath = fullfile(D.sourceDir, D.tiffSource, 'Channel01', sprintf('File%03d', tiffidx));
        tiffslices = dir(fullfile(tiffsourcePath, '*.tif'));
        tiffslices = {tiffslices(:).name}';
        
        Yt = loadtiff(fullfile(tiffsourcePath, tiffslices{1}));
        
        data.Y(size(Yt,1), size(Yt,2), nRealFrames, nVolumes) = Yt(1)*0; 
        data.Yr(size(Yt,1)*size(Yt,2)*nRealFrames, nVolumes) = Yt(1)*0;
        
        for sliceidx=1:length(tiffslices)
            tic(); tmpYt = loadtiff(fullfile(tiffsourcePath, tiffslices{sliceidx})); toc();
            data.Y(:,:,sliceidx,:) = reshape(tmpYt, [size(Yt,1) size(Yt,2) 1 nVolumes]);
        end
        data.sizY = size(data.Y);
        data.Yr(1:prod(data.sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(data.sizY(:,1:end-1)),[]);
        data.nY = min(data.Yr(:,:));
        
        fprintf('Memmapping finished for %i of %i files.\n', tiffidx, length(tiffs));
        fprintf('Size of memmaped movie is: %s\n', mat2str(data.sizY));
        
    else % ok to read in whole thing
        
        tic; Yt = loadtiff(tpath, 1, nSlices*nVolumes*nChannels); toc;

        % Only grab green channel:
        if strcmp(D.preprocessing, 'raw') && D.tefo
            Yt = Yt(:,:,1:2:end);
        end

        Y = cell(1, nVolumes);
        firstslice = 1;
        for vol=1:nVolumes
            Y{vol} = Yt(:,:,firstslice:(firstslice+nRealFrames-1));
            firstslice = firstslice+nSlices;
        end
        Y = cat(4, Y{1:end});
        if ~isa(Y, 'double'); Y = double(Y); end    % convert to double
        Y = Y -  min(Y(:));                         % make data non-negative

        data.Y = Y;
        data.sizY = size(data.Y);
        data.Yr(1:prod(data.sizY(:,1:end-1)), 1:nVolumes) = reshape(data.Y,prod(data.sizY(:,1:end-1)),[]);
        data.nY = min(data.Yr(:,:));

        fprintf('Memmapping finished for %i of %i files.\n', tiffidx, length(tiffs));
        fprintf('Size of memmaped movie is: %s\n', mat2str(data.sizY));
    end
    
    end



% 
fprintf('Done!');
toc()
end

% MAKE SUBSTACK:
%startSliceIdx = 4
%mempath = fullfile(D.nmfPath, 'memfiles');
if D.average
    mempath = D.averagePath;
else
    mempath = fullfile(D.nmfPath, 'memfiles');
end

tmpfiles = dir(fullfile(mempath, '*.mat'));
tmpfiles = {tmpfiles(:).name}';
subidxs = cell2mat(cellfun(@(x) ~isempty(strfind(x, '_substack')), tmpfiles, 'UniformOutput', 0))
files = tmpfiles(subidxs);

if isempty(files)
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

        %matpath = fullfile(mempath, sprintf('%s.mat', filename));

        %tpath = '/nas/volume1/2photon/RESDATA/20161221_JR030W/test_crossref/nmf/analysis/datastruct_002/nmf/memfiles/fov1_bar037Hz_run4_00005_crop.mat';
        %[~, filename, ext] = fileparts(tpath);

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
else
   
    fprintf('Checking TIFF substakc size...\n');
    
    for tiffidx=1:length(files)
        tpath = fullfile(mempath, files{tiffidx});
        data = matfile(tpath, 'Writable', true);
        if ~isprop(data, 'sizY')
            data.sizY = size(data.Y);
        end
        fprintf('TIFF %i of %i: size is %s.\n', tiffidx, length(files), mat2str(data.sizY));

    end
    
end
end
