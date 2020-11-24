import re
import os
import glob
import json
import traceback

import numpy as np
import pandas as pd
import cPickle as pkl

from pipeline.python.classifications import aggregate_data_stats as aggr
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def isnumber(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`, 
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    except TypeError:
        return False

    return True


# ===================================================================
# Data loading 
# ====================================================================
def create_parsed_traces_id(experiment='EXP', alignment_type='ALIGN', 
                            feature_name='FEAT', snapshot=391800):
    '''
    Common name for pupiltraces datafiles.
    '''
    fname = 'traces_%s_align-%s_%s_snapshot-%i' % (feature_name, alignment_type, experiment, snapshot)
    return fname


def load_pupil_traces_fov(animalid, session, fov, experiment, 
                        alignment_type='stimulus', snapshot=391800, 
                        feature_name='pupil_area', 
                        rootdir='/n/coxfs01/2p-data'):
    '''
    Load pupil traces for one dataset
    '''
    results=None
    params=None

    # Set output stuff
    dst_dir = os.path.join(rootdir, animalid, session, fov, 
                            'combined_%s_static' % experiment, 'facetracker')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    #print("Saving output to: %s" % dst_dir)
    
    # Create parse id
    parse_id = create_parsed_traces_id(experiment=experiment, 
                               alignment_type=alignment_type,
                               feature_name=feature_name,
                               snapshot=snapshot) 
    params_fpath = os.path.join(dst_dir, '%s_params.json' % parse_id)
    results_fpath = os.path.join(dst_dir, '%s.pkl' % parse_id)

    try:
        # load results
        with open(results_fpath, 'rb') as f:
            results = pkl.load(f)

        # load params
        with open(params_fpath, 'r') as f:
            params = json.load(f)
    except Exception as e:
        #print(e)
        return None, None    

    return results, params


def aggregate_pupil_traces(experiment, traceid='traces001', 
                feature_name='pupil_area', 
                snapshot=391800, alignment_type='stimulus',
                rootdir='/n/coxfs01/2p-data', fov_type='zoom2p0x', state='awake',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    '''
    Create AGGREGATED pupiltraces dict (from all FOVS w dlc)
    '''
    print("~~~~~ Aggregating pupil traces. ~~~~~~")

    missing_dsets=[]
    pupiltraces={}
    # Get all datasets
    sdata = aggr.get_aggregate_info(traceid=traceid, 
                                    fov_type=fov_type, state=state)
    edata = sdata[sdata['experiment']==experiment]

    for (animalid, session, fov), g in edata.groupby(['animalid', 'session', 'fov']):
        fovnum = int(fov.split('_')[0][3:])  

        datakey ='%s_%s_fov%i' % (session, animalid, fovnum)

        # for aggregrating.................
        results, params = load_pupil_traces_fov(animalid, session, fov, 
                                feature_name=feature_name,
                                alignment_type=alignment_type,
                                experiment=experiment, snapshot=snapshot, 
                                rootdir=rootdir)
        if results is None:
            missing_dsets.append(datakey)
            continue
        #### Add to dict
        pupiltraces[datakey] = results

    # Save
    traces_fname = create_parsed_traces_id(experiment=experiment, 
                            alignment_type=alignment_type,
                            feature_name=feature_name, snapshot=snapshot)
    pupil_fpath = os.path.join(aggregate_dir, 
                                'behavior-state', '%s.pkl' % traces_fname) 
    if not os.path.exists(os.path.join(aggregate_dir, 'behavior-state')):
        os.makedirs(os.path.join(aggregate_dir, 'behavior-state'))

    with open(pupil_fpath, 'wb') as f:
        pkl.dump(pupiltraces, f, protocol=pkl.HIGHEST_PROTOCOL)

    print("Aggregated pupil traces. Missing %i datasets." % len(missing_dsets))
    for m in missing_dsets:
        print(m)

    return pupiltraces


def load_pupil_traces(experiment='blobs', feature_name='pupil_area', 
                alignment_type='stimulus', snapshot=391800, 
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Load AGGREGATED pupiltraces dict (from all FOVS w dlc)
    '''

    import cPickle as pkl

    pupiltraces=None
    #### Loading existing extracted pupil data
    traces_fname = create_parsed_traces_id(experiment=experiment, 
                            alignment_type=alignment_type,
                            feature_name=feature_name, snapshot=snapshot)
#    if feature_name == 'pupil_area':
#        traces_fname = '%s_pupil_area_traces_snapshot-%i' % (experiment, snapshot)
#    else:
#        traces_fname = '%s_pupil-traces_snapshot-%i' % (experiment, snapshot)
       
    pupil_fpath = os.path.join(aggregate_dir, 
                                'behavior-state', '%s.pkl' % traces_fname) 
    
    if not os.path.exists(pupil_fpath):
        print( "NOT found: %s" % traces_fname)
        return None
    #print(pupil_fpath)

    # This is a dict, keys are datakeys
    with open(pupil_fpath, 'rb') as f:
        pupiltraces = pkl.load(f)

    print(">>>> Loaded aggregated pupil traces.")

    return pupiltraces
 
def get_aggregate_pupil_traces(experiment, feature_name='pupil_area',
                    alignment_type='stimulus', snapshot=391800, 
                    traceid='traces001', rootdir='/n/coxfs01/2p-data',
                    aggregate_dir='/n/coxfs01/2p-data/aggregate-visual-areas',
                    create_new=False):
    '''
    Load or create AGGREGATED pupiltraces dict.
    '''
    if not create_new:
        try:
            pupiltraces = load_pupil_traces(experiment=experiment, 
                            feature_name=feature_name,
                            alignment_type=alignment_type, snapshot=snapshot,
                            aggregate_dir=aggregate_dir)
            assert pupiltraces is not None, "ERROR. Re-aggregating (creating pupiltraces)"
        except Exception as e:
            traceback.print_exc()
            create_new=True

    feature_to_load = 'pupil_area' if feature_name=='pupil_fraction' else feature_name
    if create_new:
        pupiltraces = aggregate_pupil_traces(experiment, traceid=traceid,
                            feature_name=feature_to_load,alignment_type=alignment_type, 
                            snapshot=snapshot, 
                            rootdir=rootdir, aggregate_dir=aggregate_dir)
    return pupiltraces


# AGGREGATE STUFF
def create_dataframes_name(experiment, feature_name, trial_epoch, snapshot):
    '''Name for some per-trial metric calculated on traces'''

    fname = 'metrics_%s_%s_%s_snapshot-%i' % (experiment, feature_name, trial_epoch, snapshot)
    return fname

def load_pupil_dataframes(snapshot, experiment='blobs', 
                feature_name='pupil_area', trial_epoch='pre',
                aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):
    '''
    Load AGGREGATED pupil dataframes (per-trial metric).

    Returns:

    pupildata (dict)
        keys: datakeys (like MEANS dict)
        values: dataframes (pupildf, trial metrics) 
    '''
    import cPickle as pkl

    fname = create_dataframes_name(experiment, feature_name, trial_epoch, snapshot)
    pupildf_fpath = os.path.join(aggregate_dir, 
                                    'behavior-state', '%s.pkl' % fname)
    pupildata=None
    try:
        with open(pupildf_fpath, 'rb') as f:
            pupildata = pkl.load(f)
        print(">>>> Loaded aggregate pupil dataframes.")

    except Exception as e:
        print('File not found: %s' % pupildf_fpath)
    return pupildata


       
def aggregate_pupil_dataframes(pupiltraces, fname, 
                    feature_name='pupil_fraction', trial_epoch='pre',
                    in_rate=20., out_rate=20., 
                    iti_pre=1., iti_post=1., stim_dur=1.):
    '''
    Create AGGREGATED pupil dataframes (per-trial metric) - dict

    Takes pupil traces, resample, then returns dict of pupil dataframes.
    Durations are in seconds.
 
    Returns:

    pupildata (dict)
        keys: datakeys (like MEANS dict)
        values: dataframes (pupildf, trial metrics)
    '''
    print("~~~~~~~~~~~~ Aggregating pupil dataframes. ~~~~~~~~~~~")
    import cPickle as pkl

    desired_nframes = int((stim_dur + iti_pre + iti_post)*out_rate)
    iti_pre_ms=iti_pre*1000
    new_stim_on = int(round(iti_pre*out_rate))
    nframes_on = int(round(stim_dur*out_rate))

    feature_to_load = 'pupil_area' if feature_name=='pupil_fraction' else feature_name

    pupildata={}
    for dkey, ptraces in pupiltraces.items():
        #dkey = '_'.join(k.split('_')[0:-1])
        pupil_r = resample_pupil_traces(ptraces, 
                                    in_rate=in_rate, 
                                    out_rate=out_rate, 
                                    desired_nframes=desired_nframes, 
                                    feature_name=feature_to_load, #feature_name, 
                                    iti_pre_ms=iti_pre_ms)
        pupildf = get_pupil_df(pupil_r, trial_epoch=trial_epoch, 
                                new_stim_on=new_stim_on, nframes_on=nframes_on)
        if 'pupil_fraction' not in pupildf.columns:
            pupil_max = pupildf['pupil_area'].max()
            pupildf['pupil_fraction'] = pupildf['pupil_area']/pupil_max
        pupildata[dkey] = pupildf

    return pupildata
 
def get_pupil_df(pupil_r, trial_epoch='pre', new_stim_on=20., nframes_on=20.):
    '''
    Turn resampled pupil traces into reponse vectors
    
    trial_epoch : (str)
        'pre': Use PRE-stimulus period for response metric.
        'stim': Use stimulus period
        'all': Use full trial period
    
    new_stim_on: (int)
        Frame index for stimulus start (only needed if trial_epoch is 'pre' or 'stim')
        
    pupil_r : resampled pupil traces (columns are trial, frame, pupil_area, frame_int, frame_ix)
    '''
    if trial_epoch=='pre':
        pupildf = pd.concat([g[g['frame_ix'].isin(np.arange(0, new_stim_on))].mean(axis=0) \
                            for t, g in pupil_r.groupby(['trial'])], axis=1).T
    elif trial_epoch=='stim':
        pupildf = pd.concat([g[g['frame_ix'].isin(np.arange(new_stim_on, new_stim_on+nframes_on))].mean(axis=0) \
                            for t, g in pupil_r.groupby(['trial'])], axis=1).T
    else:
        pupildf = pd.concat([g.mean(axis=0) for t, g in pupil_r.groupby(['trial'])], axis=1).T
    #print(pupildf.shape)

    return pupildf


def get_aggregate_pupildfs(experiment='blobs', feature_name='pupil_area', 
                           trial_epoch='pre', alignment_type='stimulus', 
                           in_rate=20., out_rate=20., iti_pre=1., iti_post=1., stim_dur=1.,
                           snapshot=391800, 
                           aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas', 
                           create_new=False):
    '''
    Load or create AGGREGATED dit of pupil dataframes (per-trial metrics).
    (prev called load_pupil_data)
    Returns:

    pupildata (dict)
        keys: datakeys (like MEANS dict)
        values: dataframes (pupildf, trial metrics)
    '''
    if not create_new:
        try:
            pupildata = load_pupil_dataframes(snapshot, experiment=experiment, 
                                    feature_name=feature_name, 
                                    trial_epoch=trial_epoch, 
                                    aggregate_dir=aggregate_dir)
            assert pupildata is not None, "No pupil df. Creating new."
        except Exception as e:
            create_new=True

    if create_new:
        fname = create_dataframes_name(experiment, feature_name, trial_epoch, snapshot)
        pupildf_fpath = os.path.join(aggregate_dir, 'behavior-state', '%s.pkl' % fname)
        pupiltraces = get_aggregate_pupil_traces(experiment, feature_name=feature_name, 
                                    alignment_type=alignment_type, 
                                    snapshot=snapshot, 
                                    aggregate_dir=aggregate_dir,
                                    create_new=create_new)
        pupildata = aggregate_pupil_dataframes(pupiltraces, fname, 
                                    feature_name=feature_name,
                                    trial_epoch=trial_epoch,
                                    in_rate=in_rate, out_rate=out_rate, 
                                    iti_pre=iti_pre, iti_post=iti_post, stim_dur=stim_dur)

        with open(pupildf_fpath, 'wb') as f:
            pkl.dump(pupildata, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("---> Saved aggr dataframes: %s" % pupildf_fpath)

    return pupildata


def get_dlc_sources(
                    dlc_home_dir='/n/coxfs01/julianarhee/face-tracking',
                    dlc_project='facetracking-jyr-2020-01-25'):

    project_dir = os.path.join(dlc_home_dir, dlc_project)
    video_dir = os.path.join(project_dir, 'videos')
    results_dir = os.path.join(project_dir, 'pose-analysis')

    return results_dir, video_dir


def get_datasets_with_dlc(sdata, dlc_projectid='facetrackingJan25',
                        scorer='DLC_resnet50', iteration=1, shuffle=1, 
                        trainingsetindex=0, snapshot=391800,
                        dlc_home_dir='/n/coxfs01/julianarhee/face-tracking',
                        dlc_project = 'facetracking-jyr-2020-01-25'):
    # This stuff is hard-coded because we only have 1
    #### Set source/dst paths
    #dlc_home_dir = '/n/coxfs01/julianarhee/face-tracking'
    #dlc_project = 'facetracking-jyr-2020-01-25' #'sideface-jyr-2020-01-09'
    dlc_project_dir = os.path.join(dlc_home_dir, dlc_project)

    dlc_video_dir = os.path.join(dlc_project_dir, 'videos')
    dlc_results_dir = os.path.join(dlc_project_dir, 'pose-analysis') # DLC analysis output dir

    #### Training iteration info
    #dlc_projectid = 'facetrackingJan25'
    #scorer='DLC_resnet50'
    #iteration = 1
    #shuffle = 1
    #trainingsetindex=0
    videotype='.mp4'
    #snapshot = 391800 #430200 #20900
    DLCscorer = '%s_%sshuffle%i_%i' % (scorer, dlc_projectid, shuffle, snapshot)
    print("Extracting results from scorer: %s" % DLCscorer)
 
    print("Checking for existing results: %s" % dlc_results_dir)
    dlc_runkeys = list(set([ os.path.split(f)[-1].split('DLC')[0] \
                           for f in glob.glob(os.path.join(dlc_results_dir, '*.h5'))]))
    dlc_analyzed_experiments = ['_'.join(s.split('_')[0:4]) for s in dlc_runkeys]

    # Get sdata indices that have experiments analyzed
    ixs_wth_dlc = [i for i in sdata.index.tolist() 
                    if '%s_%s' % (sdata.loc[i]['datakey'], sdata.loc[i]['experiment']) in dlc_analyzed_experiments]
    dlc_dsets = sdata.iloc[ixs_wth_dlc]

    return dlc_dsets

def load_pose_data(animalid, session, fovnum, curr_exp, analysis_dir, 
                   feature_list=['pupil'],
                   alignment_type='stimulus', 
                   pre_ITI_ms=1000, post_ITI_ms=1000,
                   return_bad_files=False,
                   traceid='traces001', snapshot=391800, 
                   eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
   
    print("Loading pose data (dlc)") 
    # Get metadata for facetracker
    facemeta, missing_dlc = align_trials_to_facedata(animalid, session, fovnum, curr_exp, 
                                        alignment_type=alignment_type, 
                                        pre_ITI_ms=pre_ITI_ms, 
                                        post_ITI_ms=post_ITI_ms,
                                        return_missing=True,
                                        eyetracker_dir=eyetracker_dir)
    
    # Get pupil data
    print("Getting pose metrics by trial")
    datakey ='%s_%s_fov%i_%s' % (session, animalid, fovnum, curr_exp)  
    #pupildata, bad_files = parse_pupil_data(datakey, analysis_dir, eyetracker_dir=eyetracker_dir)
    pupildata, bad_files = calculate_pose_features(datakey, analysis_dir, 
                                            feature_list=feature_list, 
                                            snapshot=snapshot,
                                            eyetracker_dir=eyetracker_dir)
    
    if bad_files is not None and len(bad_files) > 0:
        print("___ there are %i bad files ___" % len(bad_files))
        for b in bad_files:
            print("    %s" % b)

    if return_bad_files:
        return facemeta, pupildata, missing_dlc, bad_files
    else:
        return facemeta, pupildata

# ===================================================================
# Feature extraction (traces)
# ====================================================================

def get_pose_traces(facemeta, pupildata, labels, feature='pupil', 
                    return_missing=False, verbose=False):
    '''
    Combines indices for MW trials (facemeta) with pupil traces (pupildata)
    and assigns stimulus/condition info w/ labels.
    
    '''
    print("Parsing pose data with MW")

    # Make sure we only take the included runs
    included_run_indices = labels['run_ix'].unique() #0 indexed
    mwmeta_runs = facemeta['run_num'].unique() # 1 indexed
    pupildata_runs = pupildata['run_num'].unique() # 1 indexed
    
    if 0 in included_run_indices and (1 not in mwmeta_runs): # skipped _run1
        included_run_indices1 = [int(i+2) for i in included_run_indices]
    else:
        included_run_indices1 = [int(i+1) for i in included_run_indices]

    tmpmeta = facemeta[facemeta['run_num'].isin(included_run_indices1)]
    tmppupil = pupildata[pupildata['run_num'].isin(included_run_indices1)]

    # Add stimulus config info to face data
    trial_key = pd.DataFrame({'config': [g['config'].unique()[0] \
                                            for trial, g in labels.groupby(['trial'])],
                             'trial': [int(trial[5:]) \
                                            for trial, g in labels.groupby(['trial'])]})
    facemeta = pd.concat([tmpmeta, trial_key], axis=1)
    
    # Get pos traces for each valid trial
    config_names = sorted(facemeta['config'].unique(), key=natural_keys)

    missing_trials = []
    pupiltraces = []
    for tix, (trial, g) in enumerate(facemeta.groupby(['trial'])):

        curr_config = g['config'].unique()[0] 
        # Get run of experiment that current trial is in
        run_label = g['run_label'].unique()[0]
        if run_label not in pupildata['run_label'].unique():
            if verbose:
                print("--- [trial %i] warning, run %s not found. skipping..." % (int(trial), run_label))
            missing_trials.append(trial)
            eye_values = [np.nan] 
            # continue
        
        else: 
            if feature=='pupil':
                feature_name_tmp = 'pupil_maj'
            elif 'snout' in feature:
                feature_name_tmp = 'snout_area'
            else:
                feature_name_tmp = feature

            # print("***** getting %s *****" % feature_name_tmp)
            pupil_dists_major = pupildata[pupildata['run_label']==run_label]['%s' % feature_name_tmp].values

            # Get start/end indices of current trial in run
            (eye_start, eye_end), = g[['start_ix', 'end_ix']].values

            #eye_tpoints = frames['time_stamp'][eye_start:eye_end+1]
            eye_values = pupil_dists_major[int(eye_start):int(eye_end)+1]
            
            # If all nan, get rid of this trial
            if all(np.isnan(eye_values)):
                #print("NANs: skipping", trial)
                eye_values = [np.nan]
                missing_trials.append(trial)
                #continue 
        pdf = pd.DataFrame({'%s' % feature: eye_values,
                            'config': [curr_config for _ in np.arange(0, len(eye_values))],
                            'trial': [trial for _ in np.arange(0, len(eye_values))]}, 
                            index=np.arange(0, len(eye_values)) )
        pupiltraces.append(pdf)

    pupiltraces = pd.concat(pupiltraces, axis=0) #.fillna(method='pad')  
    print("Missing %i trials total" % (len(missing_trials)), missing_trials)

    if return_missing:
        return pupiltraces, missing_trials
    else:
        return pupiltraces


# ===================================================================
# Calculate metrics (trial stats)
# ====================================================================
def calculate_pose_stats(facemeta, pupildata, labels, feature='pupil'):
    '''
    Combines indices for MW trials (facemeta) with pupil traces (pupildata)
    and assigns stimulus/condition info w/ labels.
    
    '''
    # Make sure we only take the included runs
    included_run_indices = labels['run_ix'].unique() #0 indexed
    mwmeta_runs = facemeta['run_num'].unique() # 1 indexed
    pupildata_runs = pupildata['run_num'].unique() # 1 indexed
    
    #included_run_indices1 = [int(i+1) for i in included_run_indices]
    #included_run_indices1
    
    if 0 in included_run_indices and (1 not in mwmeta_runs): # skipped _run1
        included_run_indices1 = [int(i+2) for i in included_run_indices]
    else:
        included_run_indices1 = [int(i+1) for i in included_run_indices]

    tmpmeta = facemeta[facemeta['run_num'].isin(included_run_indices1)]
    tmppupil = pupildata[pupildata['run_num'].isin(included_run_indices1)]

    # Add stimulus config info to face data
    trial_key = pd.DataFrame({'config': [g['config'].unique()[0] \
                             for trial, g in labels.groupby(['trial'])],
                  'trial': [int(trial[5:]) \
                             for trial, g in labels.groupby(['trial'])]})
    facemeta = pd.concat([tmpmeta, trial_key], axis=1)
    
    # Calculate a pupil metric for each trial
    pupilstats = get_per_trial_metrics(tmppupil, facemeta, feature_name=feature)
    
    return pupilstats


def get_per_trial_metrics(pupildata, facemeta, feature_name='pupil_maj', feature_save_name=None):
    
    
    if feature_save_name is None:
        feature_save_name = feature_name
        
    config_names = sorted(facemeta['config'].unique(), key=natural_keys)

    #pupilstats_by_config = dict((k, []) for k in config_names)
    pupilstats = []
    #fig, ax = pl.subplots()
    for tix, (trial, g) in enumerate(facemeta.groupby(['trial'])):

        # Get run of experiment that current trial is in
        run_num = g['run_num'].unique()[0]
        if run_num not in pupildata['run_num'].unique():
            #print(run_num)
            print("--- [trial %i] warning, run %s not found in pupildata. skipping..." % (trial, run_num))
            continue
        
        if feature=='pupil':
            feature_name_tmp = 'pupil_maj'
        elif 'snout' in feature_name:
            feature_name_tmp = 'snout_area'
        else:
            feature_name_tmp = feature_name
        #print("***** getting %s *****" % feature_name_tmp)
        pupil_dists_major = pupildata[pupildata['run_num']==run_num]['%s' % feature_name_tmp]

        # Get start/end indices of current trial in run
        (eye_start, eye_end), = g[['start_ix', 'end_ix']].values
        #print(trial, eye_start, eye_end)

        #eye_tpoints = frames['time_stamp'][eye_start:eye_end+1]
        eye_values = pupil_dists_major[int(eye_start):int(eye_end)+1]
        
        # If all nan, get rid of this trial
        if all(np.isnan(eye_values)):
            continue
            
        curr_config = g['config'].iloc[0]
        #curr_cond = sdf['size'][curr_config]    
        #ax.plot(eye_values.values, color=cond_colors[curr_cond])

        #print(trial, np.nanmean(eye_values))
        #pupilstats_by_config[curr_config].append(np.nanmean(eye_values))

        pdf = pd.DataFrame({'%s' % feature_save_name: np.nan if all(np.isnan(eye_valus)) else np.nanmean(eye_values),
                            'config': curr_config,
                            'trial': trial}, index=[tix])

        pupilstats.append(pdf)

    pupilstats = pd.concat(pupilstats, axis=0)
    
    return pupilstats



# ===================================================================
# Data processing 
# ====================================================================
# Data cleanup

def get_metaface_for_run(curr_src):
    '''
    Get frame times and and interpolate missing frames for 1 run
    '''
    
    try:
        run_num = int(re.search(r"_f\d+_", os.path.split(curr_src)[-1]).group().split('_')[1][1:])
        #run_num = int(re.search(r"_f\d{1}_", os.path.split(curr_src)[-1]).group().split('_')[1][1:])
    except Exception as e:
        run_num = int(re.search(r"_f\d+[a-zA-Z]_", os.path.split(curr_src)[-1]).group().split('_')[1][1:])
        #run_num = int(re.search(r"_f\d{2}_", os.path.split(curr_src)[-1]).group().split('_')[1][1:])

    # print("----- File %s.-----" % run_num)

    # Get meta data for experiment
    #errors = []
    metadata = None
    performance_info = os.path.join(curr_src, 'times', 'performance.txt')
    try:
        metadata = pd.read_csv(performance_info, sep="\t ")
        fps = float(metadata['frame_rate'])
        #print(metadata)
    except Exception as e:
        src_key = os.path.split(curr_src)[-1]
        print('***** ERROR: *****\n  Unable to load performance.txt (%s)\n  %s' % (src_key, str(e)))
        #errors.append(curr_src)
        fps = 20.0

    # Get frame info
    frame_info = os.path.join(curr_src, 'times', 'frame_times.txt')
    try:
        frame_attrs = pd.read_csv(frame_info, sep="\t ")
        #print(frame_attrs.head())
        #print("...loaded frames:", frame_attrs.shape)
    except Exception as e:
        print(e)

    frames = check_missing_frames(frame_attrs, metadata)
    #print("... adjusted for missing frames:", frames.shape)

    tif_dur_sec = frames.iloc[-1]['time_stamp'] / 60.
    print("... Full run duration: %.2f min" % tif_dur_sec)
    
    return frames


def check_missing_frames(frame_attrs, metadata, verbose=False):
    '''
    frame_attrs : pd.DataFrame with columns 
        frame_number
        sync_in1
        sync_in2
        time_stamp
        
    These are NaNs, and will be interpolated. Matters for time_stamp.
    
    Returns:
        interpdf, the interpolated dataframe
        
    '''
    if verbose:
        print("... checking for missing frames.")
    if metadata is None:
        fps = 20.0
        metadata = {'frame_period': (1./fps)}
    tmpdf = frame_attrs.copy()
    missed_ixs = [m-1 for m in np.where(frame_attrs['time_stamp'].diff() > float(metadata['frame_period']*1.5))[0]]

    if len(missed_ixs)>0:
        print("... found %i funky frame chunks: %s" % (len(missed_ixs), str(missed_ixs)))
    for mi in missed_ixs:
        # Identify duration of funky interval and how many missed frames it is:
        missing_interval = frame_attrs['time_stamp'][mi+1] - frame_attrs['time_stamp'][mi] 
        n_missing_frames = round(missing_interval/metadata['frame_period'], 0) -1
        if verbose:
            print("... interpolating %i frames" % n_missing_frames)

        add_missing = pd.DataFrame({
                      'frame_number': [np.nan for _ in np.arange(0, n_missing_frames)], #np.arange(mi+1, mi+n_missing_frames+1),
                      'sync_in1': [np.nan for _ in np.arange(0, n_missing_frames)],
                      'sync_in2': [np.nan for _ in np.arange(0, n_missing_frames)],
                      #'time_stamp': [frame_attrs['time_stamp'][mi]+(float(metadata['frame_period'])*i) \
                      #               for i in np.arange(1, n_missing_frames+1)]},
                      'time_stamp': [np.nan for _ in np.arange(0, n_missing_frames)]},
                                  index=np.linspace(mi, mi+1, n_missing_frames+2)[1:-1])

        if verbose:
            print("... adding %i frames" % add_missing.shape[0])
        df2 = pd.concat([tmpdf.iloc[:mi+1], add_missing, tmpdf.iloc[mi+1:]]) #.reset_index(drop=True)
        tmpdf = df2.copy()
        
    #df2 = tmpdf.reset_index(drop=True)

    interpdf = tmpdf.interpolate().reset_index(drop=True)
    if verbose:
        print("... frame info shape changed from %i to %i frames" % (frame_attrs.shape[0], interpdf.shape[0]))
    
    return interpdf




def align_trials_to_facedata(animalid, session, fovnum, curr_exp, 
                             alignment_type='stimulus', pre_ITI_ms=1000, post_ITI_ms=1000,
                             rootdir='/n/coxfs01/2p-data',
                            eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp',
                            verbose=False, return_missing=False,
                            blacklist=['20191018_JC113_fov1_blobs_run5']):
 
    '''
    Align MW trial events/epochs to eyetracker frames for each trial, 
    Matches eyetracker data to each "run" of a given experiment type. 
    Typically, 1 eyetracker movie for each paradigm file.
 
    epoch (str)
        'trial' : aligned frames to pre/post stimulus period, around stimulus 
        'stimulus': align frames to stimulus ON frames (no pre/post)

    blacklist (list)
        20191018_JC113_fov1_blobs_run5:  paradigm file is f'ed up?
       
    Returns:
    
    facemeta (dataframe)
        Start/end indices for each trial across all eyetracker movies in all the runs.
        These indices will be used to assign trial labels for pupl traces, in get_pose_traces()

    '''
    
    #epoch = 'stimulus_on'
    #pre_ITI_ms = 1000
    #post_ITI_ms = 1000

    datakey ='%s_%s_fov%i' % (session, animalid, fovnum)    

    # Get all runs for the current experiment
    all_runs = sorted(glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum,\
                          '%s_run*' % curr_exp)), key=natural_keys)

    #run_list = [int(os.path.split(rundir)[-1].split('_run')[-1]) for rundir in all_runs]
    run_list = [os.path.split(rundir)[-1].split('_run')[-1] for rundir in all_runs] 
    print("[%s] Found runs:" % curr_exp, run_list)
    
    # Eyetracker source files
    print("... finding movies for dset: %s" % datakey)
    facetracker_srcdirs = sorted(glob.glob(os.path.join(eyetracker_dir, 
                                    '%s*' % (datakey))), key=natural_keys)
    for si, sd in enumerate(facetracker_srcdirs):
        print(si, sd)

    # Align facetracker frames to MW trials based on time stamps
    missing_dlc=[]
    facemeta = []
    for run_num in run_list:
        print("----- File %s.-----" % run_num)

        run_numeric = int(re.findall('\d+', run_num)[0])
        if verbose:
            print("... getting MW info for %s run: %s (run_%i)" % (curr_exp, run_num, run_numeric))
        
        if '%s_%s_fov%i_%s_run%s' % (session, animalid, fovnum, curr_exp, run_numeric) in blacklist:
            continue
        
        # Get MW info for this run
        n_files = len( glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum,\
                                        '*%s_run%i' % (curr_exp, run_numeric), 'raw*', '*.tif')) )
        
        mw_file = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum,\
                                        '*%s_run%i' % (curr_exp, run_numeric), \
                                        'paradigm', 'trials_*.json'))[0]
        with open(mw_file, 'r') as f:
            mw = json.load(f)

        #trialnames = sorted(mw.keys(), key=natural_keys)
        file_ixs = np.arange(0, n_files)
        trialnames = sorted([t for t, md in mw.items() if md['block_idx'] in file_ixs \
                            and md['stimuli']['type'] != 'blank'], key=natural_keys)
        if verbose:
            print("... %i tifs in run (%i trials)" % (n_files, len(trialnames)))
       
        start_t = mw[trialnames[0]]['start_time_ms'] - mw[trialnames[0]]['iti_dur_ms']
        #end_t = mw[trialnames[0]]['end_time_ms']

        # Get corresponding eyetracker dir for run
        try:
            curr_face_srcdir = [s for s in facetracker_srcdirs if '%s_f%s_' % (curr_exp, run_num) in s][0]
            print('... Eyetracker dir: %s' % os.path.split(curr_face_srcdir)[-1])
        except Exception as e:
            print("... ERROR (%s): Unable to load run %s" % (datakey, run_num))
            traceback.print_exc()
            print('... Check for: %s|%s|fov%i -- %s_run%s \n(%s)' % (animalid, session, fovnum, curr_exp, run_num, eyetracker_dir))
            missing_dlc.append(('%s_%s_fov%i' % (session, animalid, fovnum), '%s_%s' % (curr_exp, run_num)))
            continue
 
        # Get eyetracker metadata
        faceframes_meta = get_metaface_for_run(curr_face_srcdir)

        #face_indices = {}
        for tix, curr_trial in enumerate(sorted(trialnames, key=natural_keys)):

            parafile = str(os.path.split(mw[curr_trial]['behavior_data_path'])[-1])

            # Get SI triggers for start and end of trial
            if 'retino' in curr_exp:
                trial_num = int(curr_trial)
                curr_trial_triggers = mw[str(curr_trial)]['stiminfo']['trigger_times']
                units = 1E6
            else:
                trial_num = int(curr_trial[5:])
                if alignment_type == 'trial':
                    stim_on_ms = mw[curr_trial]['start_time_ms']
                    stim_dur_ms = mw[curr_trial]['stim_dur_ms']
                    curr_trial_triggers = [stim_on_ms-pre_ITI_ms, stim_on_ms+stim_dur_ms+post_ITI_ms]
                elif alignment_type == 'stimulus':
                    stim_on_ms = mw[curr_trial]['start_time_ms']
                    stim_dur_ms = mw[curr_trial]['stim_dur_ms']
                    curr_trial_triggers = [stim_on_ms, stim_on_ms + stim_dur_ms]

                else:
                    curr_trial_triggers = [mw[curr_trial]['start_time_ms'], mw[curr_trial]['end_time_ms']]
                units = 1E3

            # Calculate trial duration in secs
            # nsecs_trial = ( (curr_trial_triggers[1] - curr_trial_triggers[0]) / units ) 
            # Get number of eyetracker frames this corresponds to
            # nframes_trial = nsecs_trial * metadata['frame_rate']

            # Get start time and end time of trial (or tif) relative to start of RUN
            trial_start_sec = (curr_trial_triggers[0] - start_t) / units
            trial_end_sec = (curr_trial_triggers[-1] - start_t) / units
            #print("Rel trial start/stop (sec):", trial_start_sec, trial_end_sec)

            # Get corresponding eyetracker frame indices for start and end time points
            eye_start = np.where(abs(faceframes_meta['time_stamp']-trial_start_sec) == (abs(faceframes_meta['time_stamp']-trial_start_sec).min()))[0][0]
            eye_end = np.where(abs(faceframes_meta['time_stamp']-trial_end_sec) == (abs(faceframes_meta['time_stamp']-trial_end_sec).min()) )[0][0]

            if verbose:
                print("Eyetracker start/stop frames:", eye_start, eye_end)
            #face_indices[trial_num] = (eye_start, eye_end)

            face_movie = '_'.join(os.path.split(curr_face_srcdir)[-1].split('_')[0:-1])
            tmpdf = pd.DataFrame({'start_ix': eye_start,
                                  'end_ix': eye_end,
                                  'trial_in_run': trial_num,
                                  'run_label': run_num,
                                  'run_num': run_numeric,
                                  'alignment_type': alignment_type,
                                  'movie': face_movie}, index=[tix])

            facemeta.append(tmpdf)
    facemeta = pd.concat(facemeta, axis=0).reset_index(drop=True)


    print("There were %i missing DLC results." % len(missing_dlc))
    for d in missing_dlc:
        print(d)
    if return_missing:
        return facemeta, missing_dlc
    return facemeta


def calculate_pose_features(datakey, analysis_dir, feature_list=['pupil'], 
                    snapshot=391800, 
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):

    '''
    Load DLC pose analysis results, extract feature for all runs of the experiment.
    Assigns face-data frames to MW trials.
    (Use to be called: parse_pose_data)

    Returns:
    
    pupildata (dataframe) 
        Contains all analyzed (and thresholded) frames for all runs.
        NaNs if no data -- no trials yet, either.

    bad_files (list)
        Runs where no pupil data was found, though we expected it.

    '''
    #print("Parsing pose data...")
    # DLC outfiles
    dlc_outfiles = sorted(glob.glob(os.path.join(analysis_dir, 
                    '%s*_%i.h5' % (datakey, snapshot))), key=natural_keys)
    #print(dlc_outfiles)
    if len(dlc_outfiles)==0:
        print("***ERROR: no files found for dlc (analysis dir:  \n%s)" % analysis_dir)
        return None, None

    # Eyetracker source files
    # print("... checking movies for dset: %s" % datakey)
    facetracker_srcdirs = sorted(glob.glob(os.path.join(eyetracker_dir, 
                                        '%s*' % (datakey))), key=natural_keys)
    print("... found %i DLC outfiles, expecting %i based on N eyetracker dirs." % (len(dlc_outfiles), len(facetracker_srcdirs)))
    
    # Check that run num is same for PARA file and DLC results
    for fd, od in zip(facetracker_srcdirs, dlc_outfiles):
        fsub = os.path.split(fd)[-1]
        osub = os.path.split(od)[-1]
        #print('names:', fsub, osub)
        try:
            face_fnum = re.search(r"_f\d+_", fsub).group().split('_')[1][1:]
            dlc_fnum = re.search(r"_f\d+DLC", osub).group().split('_')[1][1:-3]
        except Exception as e:
            #traceback.print_exc()
            face_fnum = re.search(r"_f\d+[a-zA-Z]_", fsub).group().split('_')[1][1:]
            dlc_fnum = re.search(r"_f\d+[a-zA-Z]DLC", osub).group().split('_')[1][1:-3] 
        assert dlc_fnum == face_fnum, "incorrect match: %s / %s" % (fsub, osub)

    bad_files = []
    pupildata = []
    for dlc_outfile in sorted(dlc_outfiles, key=natural_keys):
        run_num=None
        try:
            fbase = os.path.split(dlc_outfile)[-1]
            run_num = re.search(r"_f\d+DLC", fbase).group().split('_')[1][1:-3]

        except Exception as e: 
            run_num = re.search(r"_f\d+[a-zA-Z]DLC", fbase).group().split('_')[1][1:-3]
        
        assert run_num is not None, "Unable to find run_num for file: %s" % dlfile
        print("...curr run: %s [%s]" % (run_num, os.path.split(dlc_outfile)[-1]))

        # Get corresponding DLC results for movie
        #dlc_outfile = [s for s in dlc_outfiles if '_f%iD' % run_num in s][0]
        
        # Calculate some statistic from pose data
        feature_dict={}
        for feature in feature_list:
            if 'pupil' in feature:
                pupil_dists_major, pupil_dists_minor = calculate_pupil_metrics(dlc_outfile)
                   
                if pupil_dists_major is not None and pupil_dists_minor is not None:
                    pupil_areas = [np.pi*p_maj*p_min for p_maj, p_min in \
                                        zip(pupil_dists_major, pupil_dists_minor)]
                    feature_dict.update({'pupil_maj': pupil_dists_major,
                                         'pupil_min': pupil_dists_minor,
                                         'pupil_area': pupil_areas})
            elif 'snout' in feature:
                snout_areas = calculate_snout_metrics(dlc_outfile)
                feature_dict.update({'snout_area': snout_areas})
            
        if len(feature_dict.keys())==0:
            bad_files.append(dlc_outfile)
            continue
        
        nsamples = len(feature_dict[feature_dict.keys()[0]]) #pupil_dists_major)
        run_numeric = int(re.findall('\d+', run_num)[0])
        feature_dict.update({'run_label': [run_num for _ in np.arange(0, nsamples)],
                             'run_num': [run_numeric for _ in np.arange(0, nsamples)],
                             'index': np.arange(0, nsamples)}) 
        pdf = pd.DataFrame(feature_dict, index=np.arange(0, nsamples)) 
        pupildata.append(pdf)

    pupildata = pd.concat(pupildata, axis=0)
    
    print("... done parsing!") 
    return pupildata, bad_files

# body feature extraction
def get_dists_between_bodyparts(bp1, bp2, df, DLCscorer=None):

    if DLCscorer is not None:
        coords1 = [np.array([x, y]) for x, y, in zip(df[DLCscorer][bp1]['x'].values, df[DLCscorer][bp1]['y'].values)]
        coords2 = [np.array([x, y]) for x, y, in zip(df[DLCscorer][bp2]['x'].values, df[DLCscorer][bp2]['y'].values)]
    else:
        coords1 = [np.array([x, y]) for x, y, in zip(df[bp1]['x'].values, df[bp1]['y'].values)]
        coords2 = [np.array([x, y]) for x, y, in zip(df[bp2]['x'].values, df[bp2]['y'].values)]

    dists = np.array([np.linalg.norm(c1-c2) for c1, c2 in zip(coords1, coords2)])
    return dists


def calculate_pupil_metrics(dlc_outfile, filtered=False, threshold=0.99):

    bodyparts = ['pupilT', 'pupilB', 'pupilL', 'pupilR', 'cornealR']

    df = pd.read_hdf(dlc_outfile)
    if df.shape[0] < 5: # sth wrong
        return None, None
    
    DLCscorer = df.columns.get_level_values(level=0).unique()[0]
    
    filtdf = df.copy()
    filtdf = filtdf[DLCscorer][bodyparts][filtdf[DLCscorer][bodyparts] >= threshold].dropna()
    kept_ixs = filtdf.index.tolist()

    if filtered:
        pupil_dists_major = get_dists_between_bodyparts('pupilT', 'pupilB', filtdf, DLCscorer=None)
        pupil_dists_minor = get_dists_between_bodyparts('pupilL', 'pupilR', filtdf, DLCscorer=None)
    else:
        pupil_dists_major = get_dists_between_bodyparts('pupilT', 'pupilB', df, DLCscorer=DLCscorer)
        pupil_dists_minor = get_dists_between_bodyparts('pupilL', 'pupilR', df, DLCscorer=DLCscorer)

    if not filtered:
        #print("Replacing bad vals")
        replace_ixs = np.array([i for i in np.arange(0, df.shape[0]) if i not in kept_ixs])
        if len(replace_ixs) > 0:
            pupil_dists_major[replace_ixs] = np.nan
            pupil_dists_minor[replace_ixs] = np.nan

        
    return pupil_dists_major, pupil_dists_minor



def calculate_snout_metrics(dlc_outfile, filtered=False, threshold=.99999999):
    from shapely import geometry

    #bodyparts = ['snoutA', 'snoutL2', 'snoutL1', 'whiskerAL', 'whiskerP', 'whiskerAU', 'snoutU1', 'snoutU2']
    bodyparts = ['snoutA', 'snoutL2', 'snoutL1', 'whiskerAL2', 'whiskerP2', 'whiskerAU2', 'snoutU1', 'snoutU2']

    df = pd.read_hdf(dlc_outfile)
    if df.shape[0] < 5: # sth wrong
        return None, None

    DLCscorer = df.columns.get_level_values(level=0).unique()[0]

    filtdf = df.copy()
    filtdf = filtdf[DLCscorer][bodyparts][filtdf[DLCscorer][bodyparts] >= threshold].dropna()
    kept_ixs = filtdf.index.tolist()
    
    if filtered:
        xcoords = filtdf[bodyparts].xs(('x'), level=('coords'), axis=1)
        ycoords = filtdf[bodyparts].xs(('y'), level=('coords'), axis=1)
    else:
        xcoords = df[DLCscorer][bodyparts].xs(('x'), level=('coords'), axis=1)
        ycoords = df[DLCscorer][bodyparts].xs(('y'), level=('coords'), axis=1)
    
    nsamples = xcoords.shape[0]
    snout_areas = np.array([PolyArea(xcoords.iloc[i,:], ycoords.iloc[i,:]) for i in np.arange(0, nsamples)])

    if not filtered:
        replace_ixs = np.array([i for i in np.arange(0, df.shape[0]) if i not in kept_ixs])
        if len(replace_ixs) > 0:
            snout_areas[replace_ixs] = np.nan
            snout_areas[replace_ixs] = np.nan

    return snout_areas

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class struct():
    pass

def subtract_condition_mean(neuraldata, labels, included_trials):
    
    # Remove excluded trials and Calculate neural residuals
    trial_configs = pd.DataFrame(np.vstack([g['config'].iloc[0]\
                                        for trial, g in labels.groupby(['trial']) \
                                           if int(trial[5:]) in included_trials]), columns=['config']) # trials should be 1-indexed
    trial_configs = trial_configs.loc[included_trial_ixs]
    
    # Do mean subtraction for neural data
    residuals_neural = neuraldata.copy()
    for c, g in trial_configs.groupby(['config']):
        residuals_neural.loc[g.index] = neuraldata.loc[g.index] - neuraldata.loc[g.index].mean(axis=0)

    return residuals_neural


# ===================================================================
# Trace processing 
# ====================================================================
from scipy import interpolate

def resample_traces(samples, in_rate=44.65, out_rate=20.0):

    n_in_samples= len(samples)
    in_samples = samples.copy() #[rid, :] #np.array(tracef['File%03d' % curr_file][trace_type][:])
    in_tpoints = np.arange(0, n_in_samples) #len(in_samples))

    n_out_samples = round(n_in_samples * out_rate/in_rate)
    #print("N out samples: %i" % n_out_samples)

    flinear = interpolate.interp1d(in_tpoints, in_samples, axis=0)

    out_tpoints = np.linspace(in_tpoints[0], in_tpoints[-1], n_out_samples)
    out_samples = flinear(out_tpoints)
    #print("Out samples:", out_samples.shape)
    
    return out_tpoints, out_samples

def bin_pupil_traces(pupiltraces, feature_name='pupil',in_rate=20.0, out_rate=22.325, 
                          min_nframes=None, iti_pre_ms=1000):
    pupildfs = []
    if min_nframes is None:
        min_nframes = int(round(np.mean([len(g) for p, g in pupiltraces.groupby(['trial'])])))
    #print(min_nframes)
    for trial, g in pupiltraces.groupby(['trial']):
        if len(g[feature_name]) < min_nframes:
            npad = min_nframes - len(g[feature_name])
            vals = np.pad(g[feature_name].values, pad_width=((0, npad)), mode='edge')
        else:
            vals = g[feature_name].values[0:min_nframes]
        #print(len(vals))
        out_ixs, out_s = resample_traces(vals, in_rate=in_rate, out_rate=out_rate)
        currconfig = g['config'].unique()[0]
        new_stim_on = (iti_pre_ms/1E3)*out_rate #int(np.where(abs(out_ixs-stim_on) == min(abs(out_ixs-stim_on)))[0])
        pupildfs.append(pd.DataFrame({feature_name: out_s, 
                                       'stim_on': [new_stim_on for _ in np.arange(0, len(out_s))],
                                       'config': [currconfig for _ in np.arange(0, len(out_s))],
                                       'trial': [trial for _ in np.arange(0, len(out_s))]} ))
    pupildfs = pd.concat(pupildfs, axis=0).reset_index(drop=True)
    return pupildfs


def zscore_array(v):
    return (v-v.mean())/v.std()


def resample_pupil_traces(pupiltraces, in_rate=20., out_rate=20., iti_pre_ms=1000, desired_nframes=60, 
                         feature_name='pupil_area'):
    '''
    resample pupil traces to make sure we have exactly the right # of frames to match neural data
    '''
    binned_pupil = bin_pupil_traces(pupiltraces, feature_name=feature_name,
                                         in_rate=in_rate, out_rate=out_rate, 
                                         min_nframes=desired_nframes, iti_pre_ms=iti_pre_ms)
    trials_ = sorted(pupiltraces['trial'].unique())
    frames_ = np.arange(0, desired_nframes)
    pupil_trialmat = pd.DataFrame(np.vstack([p[feature_name].values for trial, p in binned_pupil.groupby(['trial'])]),
                                  index=trials_, columns=frames_)
    pupil_r = pupil_trialmat.T.unstack().reset_index().rename(columns={'level_0': 'trial', 
                                                                       'level_1': 'frame',
                                                                       0: feature_name})
    pupil_r['frame_int'] = [int(round(f)) for f in pupil_r['frame']]
    interp_frame_ixs = list(sorted(pupil_r['frame'].unique()))
    pupil_r['frame_ix'] = [interp_frame_ixs.index(f) for f in pupil_r['frame']]

    return pupil_r
    
def match_trials(neuraldf, pupiltraces, labels_all):
    '''
    make sure neural data trials = pupil data trials
    '''
    trials_with_pupil = list(pupiltraces['trial'].unique())
    trials_with_neural = list(labels_all['trial_num'].unique())
    n_pupil_trials = len(trials_with_pupil)
    n_neural_trials = len(trials_with_neural)

    labels = labels_all[labels_all['trial_num'].isin(trials_with_pupil)].copy()
    if n_pupil_trials > n_neural_trials:
        pupiltraces = pupiltraces[pupiltraces['trial'].isin(trials_with_neural)]
    elif n_pupil_trials < n_neural_trials:    
        print(labels.shape, labels_all.shape)
        neuraldf = neuraldf.loc[trials_with_pupil]
    
    return neuraldf, pupiltraces

def match_trials_df(neuraldf, pupildf, equalize_conditions=False):
    '''
    make sure neural data trials = pupil data trials
    '''
    from pipeline.python.classifications.aggregate_data_stats import equal_counts_df
    trials_with_pupil = list(pupildf['trial'].unique())
    trials_with_neural = neuraldf.index.tolist()
    n_pupil_trials = len(trials_with_pupil)
    n_neural_trials = len(trials_with_neural)

    if n_pupil_trials > n_neural_trials:
        pupildf = pupildf[pupildf['trial'].isin(trials_with_neural)]
    elif n_pupil_trials < n_neural_trials:    
        neuraldf = neuraldf.loc[trials_with_pupil]
  
    # Equalize trial numbers after all neural and pupil trials matched 
    if equalize_conditions:
        neuraldf = equal_counts_df(neuraldf)
        new_trials_neural = neuraldf.index.tolist()
        new_trials_pupil = pupildf['trial'].unique()
        if len(new_trials_neural) < len(new_trials_pupil):
            pupildf = pupildf[pupildf['trial'].isin(new_trials_neural)]
            
    return neuraldf, pupildf

def neural_trials_from_pupil_trials(neuraldf, pupildf):
    '''
    Given pupildf, with trial numbers (trial is column in pupildf),
    return the corresponding neuraldf trials.
    Also return subset of pupil df as needed.

    '''
    return None

def split_pupil_range(pupildf, feature_name='pupil_area', n_cuts=3, return_bins=False):
    '''
    n_cuts (int)
        4: use quartiles (0.25,  0.5 ,  0.75)
        3: use H/M/L (0.33, 0.66)
    '''

    bins = np.linspace(0, 1, n_cuts+1)[1:-1]
    low_bin = bins[0]
    high_bin = bins[-1]
    pupil_quantiles = pupildf[feature_name].quantile(bins)
    low_pupil_thr = pupil_quantiles[low_bin]
    high_pupil_thr = pupil_quantiles[high_bin]
    pupil_low = pupildf[pupildf[feature_name]<=low_pupil_thr].copy()
    pupil_high = pupildf[pupildf[feature_name]>=high_pupil_thr].copy()
    # Can also bin into low, mid, high
    #pupildf['quantile'] = pd.qcut(pupildf[face_feature], n_cuts, labels=False)
    
    if return_bins:
        return bins, pupil_low, pupil_high
    else:
        return pupil_low, pupil_high










# ===================================================================
# Neural trace processing (should prob go somewhere else)
# ====================================================================
def resample_neural_traces(roi_traces, labels=None, in_rate=44.65, out_rate=20.0, 
                           zscore=True, return_labels=True):

    # Create trial mat, downsampled: shape = (ntrials, nframes_per_trial)
    trialmat = pd.DataFrame(np.vstack([roi_traces[tg.index] for trial, tg in labels.groupby(['trial'])]),\
                            index=[int(trial[5:]) for trial, tg in labels.groupby(['trial'])])

    #### Bin traces - Each tbin is a column, each row is a sample 
    sample_data = trialmat.fillna(method='pad').copy()
    ntrials, nframes_per_trial = sample_data.shape

    #### Get resampled indices of trial epochs
    #print("%i frames/trial" % nframes_per_trial)
    out_tpoints, out_ixs = resample_traces(np.arange(0, nframes_per_trial), 
                                           in_rate=in_rate, out_rate=out_rate)
    
    #### Bin traces - Each tbin is a column, each row is a sample 
    df = trialmat.fillna(method='pad').copy().T
    xdf = df.reindex(df.index.union(out_ixs)).interpolate('values').loc[out_ixs]
    binned_trialmat = xdf.T
    n_tbins = binned_trialmat.shape[1]

    #### Zscore traces 
    if zscore:
        traces_r = binned_trialmat / binned_trialmat.values.ravel().std()
    else:
        traces_r = binned_trialmat.copy()
        
    # Reshape roi traces
    curr_roi_traces = traces_r.T.unstack().reset_index() # level_0=trial number, level_1=frame number
    curr_roi_traces.rename(columns={0: roi_traces.name}, inplace=True)
    
    if return_labels:
        configs_on_included_trials = [tg['config'].unique()[0] for trial, tg in labels.groupby(['trial'])]
        included_trials = [trial for trial, tg in labels.groupby(['trial'])]
        cfg_list = np.hstack([[c for _ in np.arange(0, n_tbins)] for c in configs_on_included_trials])
        curr_roi_traces.rename(columns={'level_0': 'trial', 'level_1': 'frame_interp'}, inplace=True)
        curr_roi_traces['config'] = cfg_list
        return curr_roi_traces
    else:
        return curr_roi_traces[roi_traces.name]

def resample_labels(labels, in_rate=44.65, out_rate=20):
    # Create trial mat, downsampled: shape = (ntrials, nframes_per_trial)
    trialmat = pd.DataFrame(np.vstack([tg.index for trial, tg in labels.groupby(['trial'])]),\
                            index=[int(trial[5:]) for trial, tg in labels.groupby(['trial'])])
    configs_on_included_trials = [tg['config'].unique()[0] for trial, tg in labels.groupby(['trial'])]
    included_trials = [trial for trial, tg in labels.groupby(['trial'])]

    #### Bin traces - Each tbin is a column, each row is a sample 
    sample_data = trialmat.fillna(method='pad').copy()
    ntrials, nframes_per_trial = sample_data.shape
    

    #### Get resampled indices of trial epochs
    print("%i frames/trial" % nframes_per_trial)
    out_tpoints, out_ixs = resample_traces(np.arange(0, nframes_per_trial), 
                                           in_rate=in_rate, out_rate=out_rate)
    
    #### Bin traces - Each tbin is a column, each row is a sample 
    df = trialmat.fillna(method='pad').copy().T
    xdf = df.reindex(df.index.union(out_ixs)).interpolate('values').loc[out_ixs]
    binned_trialmat = xdf.T
    n_tbins = binned_trialmat.shape[1]

    # Reshape roi traces
    curr_roi_traces = binned_trialmat.T.unstack().reset_index() # level_0=trial number, level_1=frame number
    curr_roi_traces.rename(columns={0: 'index'}, inplace=True)
    

    cfg_list = np.hstack([[c for _ in np.arange(0, n_tbins)] for c in configs_on_included_trials])
    curr_roi_traces.rename(columns={'level_0': 'trial', 'level_1': 'frame_interp'}, inplace=True)
    curr_roi_traces['config'] = cfg_list
    
    return curr_roi_traces

def roi_traces_to_trialmat(curr_roi_traces, trial_ixs):
    '''Assumes that label info in curr_roi_traces dataframe (return_labels=True, for resample_neural_traces())
    '''
    rid = [i for i in curr_roi_traces.columns if isnumber(i)][0]
    
    curr_ntrials = len(trial_ixs)
    curr_nframes = curr_roi_traces[curr_roi_traces['trial'].isin(trial_ixs)][rid].shape[0]/curr_ntrials
    trial_tmat = curr_roi_traces[curr_roi_traces['trial'].isin(trial_ixs)][rid].reshape((curr_ntrials,curr_nframes))
    
    return trial_tmat

import multiprocessing as mp
from functools import partial 

def initializer(terminating_):
    # This places terminating in the global namespace of the worker subprocesses.
    # This allows the worker function to access `terminating` even though it is
    # not passed as an argument to the function.
    global terminating
    terminating = terminating_


def apply_to_columns(df, labels, in_rate=44.65, out_rate=20, zscore=True):
    print("is MP")
    df = df.T
    curr_rois = df.columns
    
    newdf = pd.concat([resample_neural_traces(df[x], labels, in_rate=framerate, out_rate=face_framerate, 
                                             zscore=zscore, return_labels=False) for x in curr_rois])
    return newdf
    
def resample_roi_traces_mp(df, labels, in_rate=44.65, out_rate=20., zscore=True, n_processes=4):
    #cart_x=None, cart_y=None, sphr_th=None, sphr_ph=None,
    #                      row_vals=None, col_vals=None, resolution=None, n_processes=4):
    results = []
    terminating = mp.Event()
    
    df_split = np.array_split(df.T, n_processes)
    pool = mp.Pool(processes=n_processes, initializer=initializer, initargs=(terminating,))
    try:
        results = pool.map(partial(apply_to_columns, labels=labels,
                                   in_rate=in_rate, out_rate=out_rate, zscore=zscore), df_split)
        print("done!")
    except KeyboardInterrupt:
        pool.terminate()
        print("terminating")
    finally:
        pool.close()
        pool.join()
  
    print(results[0].shape)
    df = pd.concat(results, axis=1)
    print(df.shape)
    return df #results

def resample_all_roi_traces(traces, labels, in_rate=44.65, out_rate=20.):
    roi_list = traces.columns.tolist()
    configs_on_included_trials = [tg['config'].unique()[0] for trial, tg in labels.groupby(['trial'])]
    included_trials = [trial for trial, tg in labels.groupby(['trial'])]

    r_list=[]

    for ri, rid in enumerate(roi_list):
        if ri%20==0:
            print("... %i of %i cells" % (int(ri+1), len(roi_list)))
            
        # Create trial mat, downsampled: shape = (ntrials, nframes_per_trial)
        trialmat = pd.DataFrame(np.vstack([traces[rid][tg.index] for trial, tg in labels.groupby(['trial'])]),\
                                index=[int(trial[5:]) for trial, tg in labels.groupby(['trial'])])

        #### Bin traces - Each tbin is a column, each row is a sample 
        sample_data = trialmat.fillna(method='pad').copy()
        ntrials, nframes_per_trial = sample_data.shape

        #### Get resampled indices of trial epochs
        out_tpoints, out_ixs = resample_traces(np.arange(0, nframes_per_trial), 
                                               in_rate=in_rate, out_rate=out_rate)

        #### Bin traces - Each tbin is a column, each row is a sample 
        df = sample_data.T
        xdf = df.reindex(df.index.union(out_ixs)).interpolate('values').loc[out_ixs] # Interpolate resampled values
        binned_trialmat = xdf.T # should be Ntrials # Nframes
        n_trials, n_tbins = binned_trialmat.shape

        #### Zscore traces 
        zscored_neural = binned_trialmat / binned_trialmat.values.ravel().std()

        # Reshape roi traces
        cfg_list = np.hstack([[c for _ in np.arange(0, n_tbins)] for c in configs_on_included_trials])
        curr_roi_traces = zscored_neural.T.unstack().reset_index() # level_0=trial number, level_1=frame number
        curr_roi_traces.rename(columns={'level_0': 'trial', 'level_1': 'frame_ix', 0: rid}, inplace=True)
        r_list.append(curr_roi_traces)

    # Combine all traces into 1 dataframe (all frames x nrois)
    traces_r = pd.concat(r_list, axis=1)
    traces_r['config'] = cfg_list

    _, dii = np.unique(traces_r.columns, return_index=True)
    traces_r = traces_r.iloc[:, dii]
    
    return traces_r



