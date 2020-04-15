import os
import glob
import shutil
import json
import re
import sys
import optparse

import scipy.stats as spstats
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import cPickle as pkl

from pipeline.python.classifications import experiment_classes as util
from pipeline.python.utils import label_figure, natural_keys

def load_traces(animalid, session, fovnum, curr_exp, traceid='traces001',
               responsive_test='ROC', responsive_thr=0.05, response_type='dff', n_stds=2.5):
    
    # Load experiment neural data
    fov = 'FOV%i_zoom2p0x' % fovnum
    if curr_exp=='blobs':
        exp = util.Objects(animalid, session, fov, traceid=traceid)
    else:
        exp = util.Gratings(animalid, session, fov, traceid=traceid)
    
    exp.load(trace_type=response_type, update_self=True, make_equal=False)
    
    labels = exp.data.labels.copy()

    # Get stimulus config info
    sdf = exp.data.sdf
    if curr_exp == 'blobs':
        sdf = util.reformat_morph_values(sdf)
    n_conditions = len(sdf['size'].unique())

    # Get responsive cells
    responsive_cells, ncells_total = exp.get_responsive_cells(response_type=response_type,\
                                                              responsive_test=responsive_test,
                                                              responsive_thr=responsive_thr)
    traces = exp.data.traces[responsive_cells]

    return traces, labels, sdf


def traces_to_trials(traces, labels, epoch='stimulus'):
    '''
    Returns dataframe w/ columns = roi ids, rows = mean response to stim ON per trial
    Last column is config on given trial.
    '''
    s_on = int(labels['stim_on_frame'].mean())
    n_on = int(labels['nframes_on'].mean())

    roi_list = traces.columns.tolist()
    trial_list = np.array([int(trial[5:]) for trial, g in labels.groupby(['trial'])])
    if epoch=='stimulus':
        mean_responses = pd.DataFrame(np.vstack([np.nanmean(traces.iloc[g.index[s_on:s_on+n_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)
    elif epoch == 'baseline':
        mean_responses = pd.DataFrame(np.vstack([np.nanmean(traces.iloc[g.index[0:s_on]], axis=0)\
                                            for trial, g in labels.groupby(['trial'])]),
                                             columns=roi_list, index=trial_list)

    condition_on_trial = np.array([g['config'].unique()[0] for trial, g in labels.groupby(['trial'])])
    mean_responses['config'] = condition_on_trial

    return mean_responses


def get_aggregate_info(traceid='traces001', fov_type='zoom2p0x', state='awake', create_new=False,
                       visual_areas=['V1', 'Lm', 'Li'],
                         aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas',
                         rootdir='/n/coxfs01/2p-data'):
                       
    from pipeline.python.classifications import get_dataset_stats as gd

    sdata_fpath = os.path.join(aggregate_dir, 'dataset_info.pkl')
    if os.path.exists(sdata_fpath) and create_new is False:
        with open(sdata_fpath, 'rb') as f:
            sdata = pkl.load(f)
    else:
        sdata = gd.aggregate_session_info(traceid=traceid, 
                                           state=state, fov_type=fov_type, 
                                           visual_areas=visual_areas,
                                           rootdir=rootdir)
        sdata['fovnum'] = [int(re.findall(r'FOV(\d+)_', x)[0]) for x in sdata['fov']]
        with open(sdata_fpath, 'wb') as f:
            pkl.dump(sdata, f, protocol=pkl.HIGHEST_PROTOCOL)
            
    return sdata

def extract_options(options):
    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-D', '--root', action='store', dest='rootdir', default='/n/coxfs01/2p-data', 
                      help='root project dir containing all animalids [default: /n/coxfs01/2pdata]')
   
    # Set specific session/run for current animal:
    parser.add_option('-E', '--experiment', action='store', dest='experiment', default='', 
                      help="experiment name (e.g,. gratings, rfs, rfs10, or blobs)") #: FOV1_zoom2p0x)")

    parser.add_option('-e', '--epoch', action='store', dest='epoch', default='stimulus', 
                      help="trial epoch (default: stimulus)")
 
    parser.add_option('-t', '--traceid', action='store', dest='traceid', default='traces001', 
                      help="traceid (default: traces001)")
    parser.add_option('--test', action='store', dest='responsive_test', default='ROC', 
                      help="responsive test (default: ROC)")
    parser.add_option('--thr', action='store', dest='responsive_thr', default=0.05, 
                      help="responsive test thr (default: 0.05 for ROC)")
    parser.add_option('-d', '--response', action='store', dest='response_type', default='dff', 
                      help="response type (default: dff)")
    parser.add_option('--nstds', action='store', dest='nstds_above', default=2.5, 
                      help="only for test=nstds, N stds above (default: 2.5)")
    parser.add_option('--new', action='store_true', dest='create_new', default=False, 
                      help="flag to create new")
    
    parser.add_option('-X', '--exclude', action='store', dest='always_exclude', 
                      default=['20190426_JC078'],
                      help="Datasets to exclude bec incorrect or overlap")

    (options, args) = parser.parse_args(options)

    return options


# Select response filters
# responsive_test='ROC'
# responsive_thr = 0.05
# response_type = 'df'
# experiment = 'blobs'
#always_exclude = ['20190426_JC078']

def aggregate_and_save(experiment, traceid='traces001', response_type='dff', epoch='stimulus',
                       responsive_test='ROC', responsive_thr=0.05, n_stds=0.0, create_new=False,
                       always_exclude=['20190426_JC078'],
                       aggregate_dir='/n/coxfs01/julianarhee/aggregate-visual-areas'):

    #### Load mean trial info for responsive cells
    data_dir = os.path.join(aggregate_dir, 'data-stats')
    sdata = get_aggregate_info(traceid=traceid)
    #### Get DATA
    load_data = False
    data_desc = '%s_%s-%s_%s-thr-%.2f_%s' % (experiment, traceid, response_type, responsive_test, responsive_thr, epoch)
    data_outfile = os.path.join(data_dir, '%s.pkl' % data_desc)

    if not os.path.exists(data_outfile):
        load_data = True
    print(data_desc)

    if load_data or create_new:

        print("Getting data: %s" % experiment)
        print("Saving data to %s" % data_outfile)

        dsets = sdata[sdata['experiment']==experiment].copy()

        DATA = {}
        for (animalid, session, fovnum), g in dsets.groupby(['animalid', 'session', 'fovnum']):
            datakey = '%s_%s_fov%i' % (session, animalid, fovnum)
            if '%s_%s' % (session, animalid) in always_exclude:
                continue
                
            if '%s_%s' % (session, animalid) in ['20190522_JC084', '20191008_JC091']:
                continue
                
            # Load traces
            traces, labels, sdf = load_traces(animalid, session, fovnum, experiment, 
                                              traceid=traceid, response_type=response_type,
                                              responsive_test=responsive_test, 
                                              responsive_thr=responsive_thr, n_stds=n_stds)
            # Calculate mean trial metric
            mean_responses = traces_to_trials(traces, labels)

            DATA['%s' % datakey] = {'data': mean_responses,
                                    'sdf': sdf}

        # Save
        with open(data_outfile, 'wb') as f:
            pkl.dump(DATA, f, protocol=pkl.HIGHEST_PROTOCOL)
        print("Done!")

    return data_outfile

def main(options):
    opts = extract_options(options)
    experiment = opts.experiment
    traceid = opts.traceid
    response_type = opts.response_type
    responsive_test = opts.responsive_test
    responsive_thr = float(opts.responsive_thr)
    n_stds = float(opts.nstds_above)
    create_new = opts.create_new
    epoch = opts.epoch
    
    data_outfile = aggregate_and_save(experiment, traceid=traceid, response_type=response_type, epoch=epoch,
                                       responsive_test=responsive_test, n_stds=n_stds,
                                       responsive_thr=responsive_thr, create_new=create_new)
    
    print("saved data.")
   

if __name__ == '__main__':
    main(sys.argv[1:])
