function M = get_mw_info(mw_path, ntiffs)

run_fns = dir(fullfile(mw_path, '*.mat'));
if isempty(run_fns)
    
    fprintf('Failed to find corresponding MW-parsed .mat file for current acquisition %s.\n', source_dir);
    
    if exist(mw_path,'dir')
        fprintf('Found MW path at %s, but .mwk file is not yet parsed.\n', mw_path);
    end
    mw_to_si = 'unknown';
else
    run_fns = {run_fns(:).name}';
    fprintf('Found %i MW files and %i TIFFs.\n', length(run_fns), ntiffs);
    pymat_fn = run_fns{1};
    pymat = load(fullfile(mw_path, pymat_fn));
    nruns = length(double(pymat.triggers)); % Just check length of a var to see how many "runs" MW-parsing detected.
   
    fprintf('---------------------------------------------------------\n');
    if length(run_fns) == ntiffs        
        fprintf('Looks like indie runs.\n');
        mw_to_si = 'indie';
        mw_fidx = 1;
    elseif length(run_fns)==1
        fprintf('Multiple SI runs found in current MW file %s.\n', pymat_fn);
        fprintf('MW detected %i runs, and there are %i TIFFs.\n', nruns, ntiffs);
        fprintf('---------------------------------------------------------\n');
        if nruns==ntiffs % Likely, discarded/ignored TIFF files for given MW session
            mw_fidx = 1;
            mw_to_si = 'multi';
            fprintf('Found correct # of runs in MW session to match # TIFFs.\n');
            fprintf('Setting mw-fix to 1.\n');
        elseif nruns > ntiffs            
            mw_fidx = (nruns - ntiffs) + 1; % EX: TIFF file1 is really corresponding to MW 'run' 4 if ignoring first 3 TIFF files.
            mw_to_si = 'multi';
            fprintf('Looks like %i TIFF files should be ignored.\n', (nruns-ntiffs));
            fprintf('Setting MW file start index to %i.\n', mw_fidx);
        else
            fprintf('**Warning** There are fewer MW runs than TIFFs. Is MW file parsed correctly?\n');
            mw_to_si = 'unknown';
        end
    else               
        fprintf('Mismatch in num of MW files (%i .mwk found) and TIFF files (%i .tif found).\n', length(run_fns), ntiffs);
        mw_to_si = 'unknown';        
    end
end

switch mw_to_si
    case 'indie' % Each SI file goes with one MW/ARD file
        % do stuff
    case 'multi'
        % do other stuff
        pymat_fn = run_fns{1}; % There should only be 1 mwk file for multiple TIFFs (experiment reps).
        pymat = load(fullfile(mw_path, pymat_fn));
        run_names = cellstr(pymat.conditions);
        if pymat.stimtype=='bar'
            conditions = cellstr(pymat.bars);
        elseif pymat.stimtype=='grating'r
            conditions = cellstr(pymat.gratings);
        elseif pymat.stimtype=='image'
            conditions = cellstr(pymat.stim_idxs);
        else
            fprintf('No stimype detected in MW file...\n');
        end

    otherwise
        % fix stuff.
end


M.mw_path = mw_path;
M.run_fns = run_fns;
M.mw_to_si = mw_to_si;
M.mw_fidx = mw_fidx;
M.nruns = nruns;
M.ntiffs = ntiffs;
M.cond_types = conditions;
M.run_names = run_names;
M.pymat = pymat;
end