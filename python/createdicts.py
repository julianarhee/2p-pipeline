#!/usr/bin/env python2

import scipy.io as spio
import os
import uuid
import numpy as np
import cPickle as pkl
import h5py
import pandas as pd
import optparse

## optparse for user-input:
#source = '/nas/volume1/2photon/RESDATA/TEFO'
#session = '20161219_JR030W'
##experiment = 'gratingsFinalMask2'
## datastruct_idx = 1
#animal = 'R2B1'
#receipt_date = '2016-12-30'
#create_new = True 
#
# todo:  parse everything by session, instead of in bulk (i.e., all animals)...
# animals = ['R2B1', 'R2B2']
# receipt_dates = ['2016-12-20', '2016-12-30']

# Need better loadmat for dealing with .mat files in a non-annoying way:

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem

    return dict


def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)

    return elem_list


# ----------------------------------------------------------------------------
# Get DICTS that parse existing (kinda) data analysis info from .pkl or.mats
# ----------------------------------------------------------------------------

def get_animal_info(animal, receipt_date='0000-00-00', sex='female'):
    animal_info = dict()
    a_uuid = str(uuid.uuid4())
    animal_info[a_uuid] = {'id': a_uuid,\
			   'name': animal,\
			   'receipt_date': receipt_date,\
			   'sex': sex\
			 }

    return animal_info


def initiate_new_animal(sessionpath, animal, receipt_date, dictpath, fn='animal_info.pkl'):
    

   
    if not os.path.exists(dictpath):
        os.mkdir(dictpath)
    print "---------------------------------------------------------------------"
    print "Creating new dicts..."
    print "New dicts will be saved to:"
    print "PATH: ", dictpath
    print "---------------------------------------------------------------------"
    

    master = get_animal_info(animal, receipt_date)
    with open(os.path.join(dictpath, fn), 'wb') as f:
	pkl.dump(master, f, protocol=pkl.HIGHEST_PROTOCOL)
    f.close()
    
    print "Done:  Created INFO dict for %s." % animal
    
    return master



def get_session_info(a_uuid, date='0000-00-00', start_time='00:00:00', run_list=[], microscope={'scope': 'tefo', 'rev': 'rev_uuid'}):
    session_info = dict()
    s_uuid = str(uuid.uuid4())
    session_info = {'id': s_uuid,\
		    'animal':  a_uuid,\
		    'date': date,\
		    'start_time': start_time,\
		    'microscope': microscope['scope'],\
		    'microscope_rev': microscope['rev'],\
		    'runs': [str(r) for r in run_list]\
		   }

    return session_info



def populate_sessions(a_uuid, master, sessionpath, session_fn, dictpath):
   
    # Get all the meta info:
    sessionmeta_fn =  os.listdir(sessionpath)
    sessionmeta_fn = [f for f in sessionmeta_fn if 'sessionmeta' in f and f.endswith('.mat')][0]
    sessionmeta = loadmat(os.path.join(sessionpath, sessionmeta_fn))

    # Populate dict entries:
    date = sessionmeta['date']
    start_time = sessionmeta['time']
    microscope = sessionmeta['scope']
    run_list = sessionmeta['runs']

    session_info =  get_session_info(master[a_uuid], date, start_time, run_list, microscope) 

    with open(os.path.join(dictpath, session_fn), 'wb') as f:
	pkl.dump(session_info, f, protocol=pkl.HIGHEST_PROTOCOL)
    f.close()

    print "Done:  Created SESSION INFO dict."

    return session_info



def select_analysis_source(run_names, dictpath, dstruct_idxs_fn='dstruct_idxs.pkl', new_analysis=False):
    if new_analysis:
        didxs = dict((k, input('Enter datastruct no for run %s:' % k)) for k in run_names)
	print "Created dstruct-idxs for runs:"
       
	with open(os.path.join(dictpath, dstruct_idxs_fn), 'wb') as f:
            pkl.dump(didxs, f, protocol=pkl.HIGHEST_PROTOCOL)
	f.close()
    else:
        with open(os.path.join(dictpath, dstruct_idxs_fn), 'rb') as f:
            didxs = pkl.load(f)
        print "Loaded previously stored dstruct idx:"
    
    for k,v in didxs.iteritems():
	print k, v
    print "---------------------------------------------------------------------"
    
    return didxs 



def get_run_info(s_uuid, run_name, runmeta, run_num=0):

    r_uuid = str(uuid.uuid4())
    # a given run may have 1 or more tiff files associated with it.
    # for continuous stimuli (retinotopic mapping), 1 tiff = 1 trial.
    # for event-related (static stimulus shown for some x seconds), 1 tiff contains multiple trials. 
 
    if type(runmeta['file'])==dict:
        # then, only 1 file, and immediate keys are 'mw' and 'si' (i.e., don't need to index into files)
        tiffs = [(str(runmeta['file']['mw']['runName']), 0)]
    else:
        tiffs = [(str(runmeta['file'][tiffidx].mw.runName), runmeta['file'][tiffidx].mw.orderNum) for tiffidx in range(len(runmeta['file']))] 
   
    run_info = {'id': r_uuid,\
                'session': s_uuid,\
                'run_name': run_name,\
                'tiffs': tiffs,\
                'run_number': str(run_num),\
                'imaging_fov_um': str(runmeta['volumesize']),\
                'stages_um': str(runmeta['stages_um']),\
                'vol_rate': str(runmeta['volumerate'])\
              }

    return run_info


def populate_runs(source, session, session_info, run_fn, dictpath):
    
    tmprun_info = dict()
    for runidx,run in enumerate(session_info['runs']):
	if run=='gratingsFinalMask2' or run=='retinotopyFinalMask':
	    continue

#            runmeta, [] = get_datastruct_for_run(didxs, source, session, run)

	runpath = os.path.join(source, session, run)
	runmeta_fn = os.listdir(os.path.join(runpath, 'analysis', 'meta'))
	runmeta_fn = [f for f in runmeta_fn if 'meta' in f and f.endswith('.mat')][0]
	runmeta = loadmat(os.path.join(runpath, 'analysis', 'meta', runmeta_fn))
	
	# todo:  FIX createMetaStruct.m s.t. new fields are saved:
	runmeta['volumesize'] = [500, 500, 210]
	runmeta['volumerate'] = 4.11
	runmeta['stages_um'] = [12568, 52037, 59050] # MANUAL ENTRY (not incorp. in SI & pipeline yet)
	
	tmprun_info[run] = get_run_info(session_info['id'], run, runmeta, runidx)
     
    # Turn run_info into sth that can be indexed by r_uuid:
    run_info = dict()
    for run in tmprun_info.keys():
	r_uuid = tmprun_info[run]['id']
	run_info[r_uuid] = dict((k, v) for k,v in tmprun_info[run].iteritems())

    with open(os.path.join(dictpath, run_fn), 'wb') as f:
	pkl.dump(run_info, f, protocol=pkl.HIGHEST_PROTOCOL)
    f.close()

    print "Done:  Created RUNS dict."
   
    return run_info






def get_trials(run_info, runmeta, outputpath):

    trials = []
    
    # Get associated TIFFs in run:
    tiffs = run_info['tiffs']

    # get all trial info:
    if runmeta['stimType']=='bar':
        # just need to count each file in current run.
        tiffs.sort(key=lambda x: x[1])
        for tiff in tiffs:
            # trials will be a dict():  
            # {'left': 'trial1_uuid', 'right': 'trial2_uuid', 'top1': 'trial3_uuid', 'top2': 'trial4_uuid'...}
            # trials[tiff[1]] = str(uuid.uuid4()) # trials[x] = uuuid - x is trial no in file 
            trials.append((tiff[1], str(uuid.uuid4()))
)
    else:
        # need to cycle into nested trialstuct and get all trials:    
        trialstruct = loadmat(os.path.join(outputpath, 'stimReps.mat')) # mat structs saved as stuct or '-v7.3' need to be read with h5py (just resaved without '-v7.3' handles in psth.m to avoid this)
        stims = dir(trialstruct['slice'][10].info) # slice index does not matter here, since will be the same for all slices.
        stimnames = [n for n in stims if 'stim' in n]
        for stim in stimnames:           
            for tidx in range(len(evl("trialstruct['slice'][10].info.%s" % stim))):
                t_uuid = str(uuid.uuid4())
                trialidx_in_run = eval("trialstruct['slice'][10].info.%s[%i].trialIdxInRun" % (stim, tidx)) 
                trials.append((trialidx_in_run, t_uuid))
                
    trials.sort(key=lambda x: x[0])

    return trials


def populate_trials(s_uuid, runs, didxs, source, session, dictpath, trialinfo_fn='trial_info.pkl'):

    trial_info = dict((k, dict()) for k in runs.keys())
    trial_list = dict()
    for r_uuid in runs.keys():
	curr_run_info = runs[r_uuid]
	
	# Get datastruct info for current run/file:
	runmeta, dstruct = get_datastruct_for_run(didxs, source, session, runs[r_uuid]['run_name'])
	outputpath = dstruct['outputDir']
	trial_list[r_uuid] = get_trials(curr_run_info, runmeta, outputpath)

	# Save to .PKL:
	triallist_fn = 'trial_list_%s.pkl' % s_uuid
	with open(os.path.join(dictpath, triallist_fn), 'wb') as f:
	    pkl.dump(trial_list, f, protocol=pkl.HIGHEST_PROTOCOL)
	f.close()

	# Get trial info:
	# ----------------------------------------------------------------------------
	tinfo_fn = 'trial_info.pkl'
	runpath = os.path.join(source, session, runs[r_uuid]['run_name'])
	print "Loading parsed MW trial info from: %s" % runpath
	with open(os.path.join(runpath, 'mw_data', tinfo_fn), 'rb') as f:
	    tinfo = pkl.load(f)

	# trial parsing in SI ignores the 1st trial, so 2nd trial is the start:
	trial_list[r_uuid].sort(key=lambda x: x[0])
	for t in trial_list[r_uuid]:
	    t_uuid = t[1]
	    trial_info[r_uuid][t_uuid] = dict()
	    trial_info[r_uuid][t_uuid]['id'] = t_uuid #trial_list[t-1][1]
	    trial_info[r_uuid][t_uuid]['run'] = curr_run_info['id'] 
	    if dstruct['stimType']=='bar':
		trialnum = t[0]    
	    else:
		trialnum = t[0]+1    # Since 1st trial in event-related expmts are ignored

	    trial_info[r_uuid][t_uuid]['start_time_ms'] = tinfo[trialnum]['start_time_ms']
	    trial_info[r_uuid][t_uuid]['end_time_ms'] = tinfo[trialnum]['end_time_ms']
	    trial_info[r_uuid][t_uuid]['stimuli'] = tinfo[trialnum]['stimuli']
	    trial_info[r_uuid][t_uuid]['stim_on_times'] = tinfo[trialnum]['stim_on_times']
	    trial_info[r_uuid][t_uuid]['stim_off_times'] = tinfo[trialnum]['stim_off_times']
	    trial_info[r_uuid][t_uuid]['idx_in_run'] = t[0]

	del tinfo

	# Save to .PKL:	     
	with open(os.path.join(dictpath, trialinfo_fn), 'wb') as f:
	    pkl.dump(trial_info, f, protocol=pkl.HIGHEST_PROTOCOL)
	f.close()
	print "Done:  Created TRIAL INFO dict."

    return trial_info



def get_cell_info(run_info, runmeta, dstruct):
#    nrois = dstruct['nRois'] # todo:  this assumes that ALL runs have the same ROIs (which we want eventually, but need alignment for)
    maskmat = h5py.File(dstruct['maskarrayPath']) # todo: save path to 3D-coord masks (saved as cell array in matlab?)
    tmpmask = np.empty(maskmat['masks'].shape)
    tmpmask = maskmat['masks'].value
    masks = tmpmask.T
    del tmpmask
    volsize = runmeta['volumeSizePixels']

    #nrois = max(nroistmp)
    nrois = masks.shape[1]        
    maskarray = dict((roi, dict()) for roi in range(nrois))
    for roi in range(nrois):
        # maskarray[roi] = np.reshape(masks[:,roi], volsize, order='F')
        # Make sparse, bec otherwise too huge:
        maskarray[roi] = pd.SparseArray(masks[:,roi])
        # To return full 3d mask:  
        # mask = np.reshape(maskarray[roi].to_dense(), [x, y, z], order='F')
    
    cell_info = dict.fromkeys([i for i in maskarray.keys()], dict())
    tmp = dict((k, {'id': str(uuid.uuid4())}) for k,v in cell_info.iteritems())
    cell_info.update(tmp)
    for k in cell_info.keys():
        cell_info[k].update(run=run_info['id'])
        cell_info[k].update(mask=maskarray[k])


    return cell_info


def get_datastruct_for_run(didxs, source, session, run_name):
    runpath = os.path.join(source, session, run_name) #runs[curr_run_id]['run_name'])
    runmeta_fn = os.listdir(os.path.join(runpath, 'analysis', 'meta'))
    runmeta_fn = [f for f in runmeta_fn if 'meta' in f and f.endswith('.mat')][0]
    runmeta = loadmat(os.path.join(runpath, 'analysis', 'meta', runmeta_fn))

    datastruct_idx = didxs[run_name] #[runs[curr_run_id]['run_name']]

    datastruct = 'datastruct_%03d' % datastruct_idx
    dstruct_fn = os.listdir(os.path.join(runpath,'analysis', datastruct))
    dstruct_fn = [f for f in dstruct_fn if f.endswith('.mat')][0]

    dstruct = loadmat(os.path.join(runpath, 'analysis', datastruct, dstruct_fn))

    return runmeta, dstruct               


def get_timecourse(cells, r_uuid, trial_info, runs, dstruct, channel=1, trial_id=''):

    #cellidx = cells[c_uuid]['cell_idx']
 
    if len(runs[r_uuid]['tiffs']) > 1:
        # Get correct TIFF for chosen trial/run:
        file_idx_in_run = trial_info[r_uuid][trial_id]['idx_in_run']
        print "File IDX: ", file_idx_in_run
 
        # Load 3d traces:
        tracestruct = loadmat(os.path.join(dstruct['tracesPath'], dstruct['traceNames3D'][file_idx_in_run-1]))
    else:
        tracestruct = loadmat(os.path.join(dstruct['tracesPath'], dstruct['traceNames3D'][0]))

    rawmat = tracestruct['rawTraces']
    print "N cells in dict: ", len(cells)
    print "Size rawmat: ", rawmat.shape

    #if trial is True:
    if len(trial_id)==0:
        parse_trials = False
    else:
        parse_trials = True

    if dstruct['stimType']=='bar':
        if parse_trials is True:
            data = dict()
            data = dict((k, rawmat[:, v['cell_idx']]) for k,v in cells.iteritems())  # rawmat[:, cellidx]
        else:
            data = dict((k, rawmat[:, v['cell_idx']]) for k,v in cells.iteritems())  # rawmat[:, cellidx]

    else:
        # only get part of full file trace that is trial:
        # TO DO FIX THIS.
        print "Need to fix event-related trial traces..." 
    
       
    return data


def main():

#    # optparse for user-input:
#    source = '/nas/volume1/2photon/RESDATA/TEFO'
#    session = '20161219_JR030W'
#    #experiment = 'gratingsFinalMask2'
#    # datastruct_idx = 1
#    animal = 'R2B1'
#    receipt_date = '2016-12-30'
#    create_new = False #True 
#    get_time = True
    
    parser = optparse.OptionParser()
    parser.add_option('-S', '--source', action='store', dest='source', default='', help='source dir (parent of session dir)')
    parser.add_option('-s', '--session', action='store', dest='session', default='', help='session dir (format: YYYMMDD_ANIMALID')
    parser.add_option('-a', '--animal', action='store', dest='animal', default='', help='animal sample ID (ex: R2B1)')
    parser.add_option('-r', '--receipt', action='store', dest='receipt_date', default='9999-99-99', help='receipt date (format: YYYY-MM-DD)')
    parser.add_option('--new', action='store_true', dest='create_new', default=False, help='create new uuid indexed dicts')
    parser.add_option('--dstruct', action='store_true', dest='new_analysis', default=False, help='Enter new dstruct analysis nos.')
    parser.add_option('-P', '--dpath', action='store', dest='dictpath', default='', help='path to save dicts')

    parser.add_option('--timecourse', action='store_true', dest='get_timecourse', default=False, help='extract time courses for each roi')
    parser.add_option('--trials', action='store_true', dest='do_trials', default=False, help='parse time-courses for trials')

    (options, args) = parser.parse_args()

    source = options.source
    session = options.session
    animal = options.animal
    receipt_date = options.receipt_date
    create_new = options.create_new
    new_analysis = options.new_analysis
    dictpath = options.dictpath
    extract_timecourse = options.get_timecourse
    do_trials = options.do_trials

    # ----------------------------------------------------------------------------
    # TEST OUTPUT: 
    # ----------------------------------------------------------------------------
    

    sessionpath = os.path.join(source, session)
 
    # Save dicts to session path:
    if len(dictpath)==0:
        dictpath = os.path.join(sessionpath, 'dicts')
 

    # 1.  Animal Info:
    # ----------------------------------------------------------------------------
    # Animal info (and subsequent sections) will be defined by whether this
    # script should be run on a per-animal or per-animal-per-session basis.
    animalinfo_fn = 'animal_info.pkl' 
    if create_new:
        master = initiate_new_animal(sessionpath, animal, receipt_date, dictpath, fn=animalinfo_fn)
    else:
	print "---------------------------------------------------------------------"
	print "Loading saved dicts..."
	print "Saved dicts from:"
	print "PATH: ", dictpath
	print "---------------------------------------------------------------------"

        with open(os.path.join(dictpath, animalinfo_fn), 'rb') as f:
            master = pkl.load(f)
        print "Loaded animal-info struct from: %s" % os.path.join(dictpath, animalinfo_fn)
    

    print "ANIMAL a_uuid: ",  master.keys()
    print "---------------------------------------------------------------------"



    # 2.  Session Info: 
    # ----------------------------------------------------------------------------
    # There can (but need not be) multiple sessions per animal. Each session can 
    # be identified by some combination of date and microscope type.

    a_uuid = [k for k in master.keys() if master[k]['name']==animal][0]

    session_fn = 'session_info_%s.pkl' % animal
    if create_new:
        session_info = populate_sessions(a_uuid, master, sessionpath, session_fn, dictpath)
    else:
        # LOAD animal's session_info:
        with open(os.path.join(dictpath, session_fn), 'rb') as f:
            session_info = pkl.load(f)
        print "Loaded session-info struct from: %s" % os.path.join(dictpath, session_fn)
    
    print "SESSION s_uuid:", session_info['id']
    print "---------------------------------------------------------------------"



    # 3.  Run Info:
    # ------------------------------------------------------------------------
    # There can (but need not be) multiple runs per session. A given run may have
    # a single-trial or multi-trial format.

    # Load a specific analysis info/structs (group of mat files indexed by
    # datastruct_idx for specific run:

    run_names = session_info['runs']
    dstruct_idxs_fn = 'dstruct_idxs.pkl'
    didxs = select_analysis_source(run_names, dictpath, dstruct_idxs_fn, new_analysis)

    run_fn = 'run_info_%s.pkl' % session_info['id']

    if create_new:
        run_info = populate_runs(source, session, session_info, run_fn, dictpath)
    else:
        # LOAD:
        with open(os.path.join(dictpath, run_fn), 'rb') as f:
            run_info = pkl.load(f)
    
        print "Loaded run-info stuct from %s:" % os.path.join(dictpath, run_fn)
    
    print 'RUNS: r_uuids:'
    for ridx,run in enumerate(run_info.keys()):
        print ridx, run_info[run]['run_name'], run
        print "---------------------------------------------------------------------"



    # 4.  Cell / Trial Info:
    # ----------------------------------------------------------------------------
    # All runs should have the same ROIs, so that comparing cell activity across
    # different runs is possible.  However, for now, am assuming that we're only
    # grabbing cell info (time courses, trials, etc.) within a given run, so we
    # grab all that info by providing some run.

    trialinfo_fn = 'trial_info.pkl'
    if create_new:
        trial_info = populate_trials(session_info['id'], run_info, didxs, source, session, dictpath, trialinfo_fn=trialinfo_fn)
    else:
        # LOAD:
        with open(os.path.join(dictpath, trialinfo_fn), 'rb') as f:
            trial_info = pkl.load(f)
        print "Loaded trial-info struct from: %s" % os.path.join(dictpath, trialinfo_fn)
        print trial_info
        print "---------------------------------------------------------------------"




    # Get cell info:
    # ----------------------------------------------------------------------------

    #cellinfo_fn = 'cell_info.pkl'
    print "Getting ROIs for %i runs..." % len(run_info.keys())

    for r_uuid in run_info.keys():
        print "Getting ROIs from run %s." % r_uuid
        cell_info = dict() #dict((k, dict()) for k in runs.keys())

        cellinfo_fn = 'cell_info_%s.pkl' % r_uuid # runs[curr_run_id]['run_name']
        if create_new:
            curr_run_info = run_info[r_uuid]

            runmeta, dstruct = get_datastruct_for_run(didxs, source, session, run_info[r_uuid]['run_name'])

            tmpcell_info = get_cell_info(curr_run_info, runmeta, dstruct)
            for cell in tmpcell_info.keys():
                c_uuid = tmpcell_info[cell]['id']
                cell_info[c_uuid] = dict((k, v) for k,v in tmpcell_info[cell].iteritems())
                cell_info[c_uuid]['cell_idx'] = cell

            with open(os.path.join(dictpath, cellinfo_fn), 'wb') as f:
                pkl.dump(cell_info, f, protocol=pkl.HIGHEST_PROTOCOL)
            f.close()

            del tmpcell_info
            
            print "Done:  Created CELLS dict."
        else:
            with open(os.path.join(dictpath, cellinfo_fn), 'rb') as f:
                cell_info = pkl.load(f)
            print "Loaded cell-info dict: %s" % cellinfo_fn


    # ----------------------------------------------------------------------------
    # LOAD DICTS AND EXTRACT CELL INFO:
    # ----------------------------------------------------------------------------
   
    if extract_timecourse:
        do_trials = options.do_trials
        for r_uuid in run_info.keys():
        
            timecourses_fn = 'timecourses_%s.pkl' % r_uuid   
            if create_new:
                # Takes FOREVER -- need to store this differently mebe:
                cellinfo_fn = 'cell_info_%s.pkl' % r_uuid #runs[r_uuid]['run_name']
                with open(os.path.join(dictpath, cellinfo_fn), 'rb') as f:
                    cell_info = pkl.load(f)
                    if not type(cell_info)==dict:
                        print "Cells in %s, not a dict." % cellinfo_fn
                        continue
                print "Getting traces for %i cells..." % len(cell_info)
                print min([int(v['cell_idx']) for c,v in cell_info.iteritems()])
                print max([int(v['cell_idx']) for c,v in cell_info.iteritems()])

                runmeta, dstruct = get_datastruct_for_run(didxs, source, session, run_info[r_uuid]['run_name'])
                trial_list = trial_info[r_uuid].keys()
               
                timecourses = dict((t_uuid, get_timecourse(cell_info, r_uuid, trial_info, run_info, dstruct, channel=1, trial_id=t_uuid)) for t_uuid in trial_list)
                with open(os.path.join(dictpath, timecourses_fn), 'wb') as f:
                    pkl.dump(timecourses, f, protocol=pkl.HIGHEST_PROTOCOL)
                f.close()
                
                print "Done:  Created TIMECOURSE dict."

            else:
                # LOAD TRACES:
                with open(os.path.join(dictpath, timecourses_fn), 'rb') as f:
                    timecourses = pkl.load(f)


    # PER-CELL-METRICS.
    # ------------------------------------------------------------------------




if __name__ == "__main__":
    main()






