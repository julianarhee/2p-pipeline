function sessionmeta = get_session_info(sessionpath, tefo)

    runpaths = uipickfiles('FilterSpec', sessionpath)

    runlist = {};
    metapaths = {};
    run_start_times = zeros(1,length(runpaths));
    for ridx = 1:length(runpaths)
        [fp, fn, ~] = fileparts(runpaths{ridx});
        runlist{end+1} = fn;

        metapath = dir(fullfile(runpaths{ridx}, 'analysis', 'meta', 'meta*.mat'));
        metapaths{end+1} = fullfile(runpaths{ridx}, 'analysis', 'meta', metapath(1).name);
    end
    
    
    for ridx=1:length(runlist)
        runmeta = load(metapaths{ridx});
        runnames = fieldnames(runmeta.file(1).mw.pymat);
        trigger_start_times = zeros(1, length(runnames));
        for subidx=1:length(runnames)
            trigger_start_times(subidx) = runmeta.file(1).mw.pymat.(runnames{subidx}).MWtriggertimes(1);
        end
        run_start_times(ridx) = min(trigger_start_times);
    end
    
    [src, session, ~] = fileparts(sessionpath);
    spts = strsplit(session, '_'); % DATE_ANIMALID

    sessionmeta.runs = runlist;
    sessionmeta.run_times = run_start_times;
    sessionmeta.time = min(run_start_times)/1E3; % put into msec
    sessionmeta.date = datestr(datenum(spts{1},'yyyyMMdd'),'yyyy-MM-dd');
    sessionmeta.animal = spts{2};
    
    % All runs in sesh should have the same scope info:
    scope = struct();
    if tefo %runmeta.tefo
        scope.scope = 'tefo';
        phase1time = '20170401';
        if datetime(spts{1}, 'InputFormat', 'yyyyMMdd') < datetime(phase1time, 'InputFormat', 'yyyyMMdd')
            scope.rev = 'rev1';
        else
            scope.rev = 'rev2';
        end
    else
        scope.scope = 'res';
        if datetime(spts{1}, 'InputFormat', 'yyyyMMdd') < datetime(phase1time, 'InputFormat', 'yyyyMMdd')
            scope.rev = 'rev1';
        else
            scope.rev = 'rev2';
        end
    end
        
    sessionmeta.scope = scope;
    
    
    struct_name = sprintf('sessionmeta_%s.mat', session);
    structpath = fullfile(sessionpath, struct_name);
    save(structpath, '-struct', 'sessionmeta')
    

end
