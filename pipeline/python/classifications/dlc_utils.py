import re
import os
import glob
import json
import traceback

import numpy as np
import pandas as pd



def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def get_datasets_with_dlc(sdata):
    # This stuff is hard-coded because we only have 1
    #### Set source/dst paths
    dlc_home_dir = '/n/coxfs01/julianarhee/face-tracking'
    dlc_project = 'facetracking-jyr-2020-01-25' #'sideface-jyr-2020-01-09'
    dlc_project_dir = os.path.join(dlc_home_dir, dlc_project)

    dlc_video_dir = os.path.join(dlc_home_dir, dlc_project, 'videos')
    dlc_results_dir = os.path.join(dlc_project_dir, 'pose-analysis') # DLC analysis output dir

    #### Training iteration info
    dlc_projectid = 'facetrackingJan25'
    scorer='DLC_resnet50'
    iteration = 1
    shuffle = 1
    trainingsetindex=0
    videotype='.mp4'
    snapshot = 391800 #430200 #20900
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

def load_pose_data(animalid, session, fovnum, curr_exp, analysis_dir, feature_list=['pupil'],
                   epoch='stimulus_on', pre_ITI_ms=1000, post_ITI_ms=1000,
                   traceid='traces001', eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):
   
    print("Loading pose data (dlc)") 
    # Get metadata for facetracker
    facemeta = align_trials_to_facedata(animalid, session, fovnum, curr_exp, 
                                        epoch=epoch, pre_ITI_ms=pre_ITI_ms, post_ITI_ms=post_ITI_ms,
                                        eyetracker_dir=eyetracker_dir)
    
    # Get pupil data
    print("Getting pose metrics by trial")
    datakey ='%s_%s_fov%i_%s' % (session, animalid, fovnum, curr_exp)  
    #pupildata, bad_files = parse_pupil_data(datakey, analysis_dir, eyetracker_dir=eyetracker_dir)
    pupildata, bad_files = parse_pose_data(datakey, analysis_dir, feature_list=feature_list, 
                                            eyetracker_dir=eyetracker_dir)
    
    if bad_files is not None and len(bad_files) > 0:
        print("___ there are %i bad files ___" % len(bad_files))
        for b in bad_files:
            print("    %s" % b)

    return facemeta, pupildata


#def calculate_pupil_stats(facemeta, pupildata, labels):
def get_pose_traces(facemeta, pupildata, labels, feature='pupil', verbose=False):
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

    pupiltraces = pd.concat(pupiltraces, axis=0).fillna(method='pad')  
    print("Missing %i trials total" % (len(missing_trials)))

    return pupiltraces



#def calculate_pupil_stats(facemeta, pupildata, labels):
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
                             epoch='stimulus_on', pre_ITI_ms=1000, post_ITI_ms=1000,
                             rootdir='/n/coxfs01/2p-data',
                            eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp',
                            blacklist=['20191018_JC113_fov1_blobs_run5'], verbose=False):
    
    '''
    Align MW trial events/epochs to eyetracker data for each trial, 
        i.e., matches eyetracker data to each "run" of a given experiment type.
        Typically, 1 eyetracker movie for each paradigm file.
    
    Returns:
        dataframe of start/end indices for each trial across all eyetracker movies in all the runs.
    '''
    
    #epoch = 'stimulus_on'
    #pre_ITI_ms = 1000
    #post_ITI_ms = 1000

    datakey ='%s_%s_fov%i_%s' % (session, animalid, fovnum, curr_exp)    

    # Get all runs for the current experiment
    all_runs = sorted(glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum,\
                          '%s*_run*' % curr_exp)), key=natural_keys)

    #run_list = [int(os.path.split(rundir)[-1].split('_run')[-1]) for rundir in all_runs]
    run_list = [os.path.split(rundir)[-1].split('_run')[-1] for rundir in all_runs] 
    print("[%s] Found runs:" % curr_exp, run_list)
    
    # Eyetracker source files
    print("... finding movies for dset: %s" % datakey)
    facetracker_srcdirs = sorted(glob.glob(os.path.join(eyetracker_dir, '%s*' % (datakey))), key=natural_keys)
    for si, sd in enumerate(facetracker_srcdirs):
        print(si, sd)

    # Align facetracker frames to MW trials based on time stamps
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
                                        '*%s*_run%i' % (curr_exp, run_numeric), 'raw*', '*.tif')) )
        
        mw_file = glob.glob(os.path.join(rootdir, animalid, session, 'FOV%i*' % fovnum,\
                                        '*%s*_run%i' % (curr_exp, run_numeric), \
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
            curr_face_srcdir = [s for s in facetracker_srcdirs if '_f%s_' % run_num in s][0]
            print('... Eyetracker dir: %s' % os.path.split(curr_face_srcdir)[-1])
        except Exception as e:
            print("... ERROR (%s): Unable to load run %s" % (datakey, run_num))
            traceback.print_exc()
            print(facetracker_srcdirs)
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
                if epoch == 'trial_alignment':
                    stim_on_ms = mw[curr_trial]['start_time_ms']
                    stim_dur_ms = mw[curr_trial]['stim_dur_ms']
                    curr_trial_triggers = [stim_on_ms - pre_ITI_ms, stim_on_ms + stim_dur_ms + post_ITI_ms]
                elif epoch == 'stimulus_on':
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
                                  'movie': face_movie}, index=[tix])

            facemeta.append(tmpdf)
    facemeta = pd.concat(facemeta, axis=0).reset_index(drop=True)

    return facemeta


def parse_pose_data(datakey, analysis_dir, feature_list=['pupil'], 
                    eyetracker_dir='/n/coxfs01/2p-data/eyetracker_tmp'):

    '''
    Loads DLC pose analysis results, extracts some feature of the behavior for all runs of the experiment.
    Assigns face-data frames to MW trials.
    
    Returns:
        dataframe that contains all analyzed (and thresholded) frames for all runs.
    '''
    #print("Parsing pose data...")
    # DLC outfiles
    dlc_outfiles = sorted(glob.glob(os.path.join(analysis_dir, '%s*.h5' % datakey)), key=natural_keys)
    #print(dlc_outfiles)
    if len(dlc_outfiles)==0:
        print("***ERROR: no files found for dlc (analysis dir:  \n%s)" % analysis_dir)
        return None, None

    # Eyetracker source files
    # print("... checking movies for dset: %s" % datakey)
    facetracker_srcdirs = sorted(glob.glob(os.path.join(eyetracker_dir, '%s*' % (datakey))), key=natural_keys)
    print("... found %i DLC outfiles, expecting %i based on found eyetracker dirs." % (len(dlc_outfiles), len(facetracker_srcdirs)))
    
    # Check that run num is same for PARA file and DLC results
    for fd, od in zip(facetracker_srcdirs, dlc_outfiles):
        fsub = os.path.split(fd)[-1]
        osub = os.path.split(od)[-1]
        #print('names:', fsub, osub)
        try:
            face_fnum = re.search(r"_f\d+_", fsub).group().split('_')[1][1:]
            dlc_fnum = re.search(r"_f\d+DLC", osub).group().split('_')[1][1:-3]
            #face_fnum = int(re.search(r"_f\d{1}_", fsub).group().split('_')[1][1:])
            #dlc_fnum = int(re.search(r"_f\d{1}D", osub).group().split('_')[1][1:-1])
        except Exception as e:
            #traceback.print_exc()
            face_fnum = re.search(r"_f\d+[a-zA-Z]_", fsub).group().split('_')[1][1:]
            dlc_fnum = re.search(r"_f\d+[a-zA-Z]DLC", osub).group().split('_')[1][1:-3] 
            #face_fnum = int(re.search(r"_f\d{2}_", fsub).group().split('_')[1][1:])
            #dlc_fnum = int(re.search(r"_f\d{2}D", osub).group().split('_')[1][1:-1])
        assert dlc_fnum == face_fnum, "incorrect match: %s / %s" % (fsub, osub)

    bad_files = []
    pupildata = []
    for dlc_outfile in sorted(dlc_outfiles, key=natural_keys):
        run_num=None
        try:
            # run_num = int(re.search(r"_f\d{1}D", os.path.split(dlc_outfile)[-1]).group().split('_')[1][1:-1])
            fbase = os.path.split(dlc_outfile)[-1]
            run_num = re.search(r"_f\d+DLC", fbase).group().split('_')[1][1:-3]

        except Exception as e: 
            run_num = re.search(r"_f\d+[a-zA-Z]DLC", fbase).group().split('_')[1][1:-3]
        
        assert run_num is not None, "Unable to find run_num for file: %s" % dlfile
        #assert np.isnan(run_numeric)==False, "Unable to find run_num for file: %s" % dlfile
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
