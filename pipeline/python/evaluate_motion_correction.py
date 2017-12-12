
import os
import h5py
import json


run = 'scenes'
pid_hash = ''

def main(options):
    # -------------------------------------------------------------
    # Set basename for files created containing meta/reference info:
    # -------------------------------------------------------------
    raw_simeta_basename = 'SI_%s' % run #functional_dir
    run_info_basename = '%s' % run #functional_dir
    pid_info_basename = 'pids_%s' % run

    # -------------------------------------------------------------
    # Set paths:
    # -------------------------------------------------------------
    acquisition_dir = os.path.join(rootdir, animalid, session, acquisition)
    pidinfo_path = os.path.join(acquisition_dir, run 'processed', '%s.json' % pid_info_basename)
    runmeta_path = os.path.join(acquisition_dir, run, '%s.json' % run_info_basename)

    # -------------------------------------------------------------
    # Load PID:
    # -------------------------------------------------------------
    with open(pidinfo_path, 'r') as f:
        pdict = json.load(f)

    process_id = [p for p in pdict.keys() if pdict[p]['pid_hash'] == pid_hash][0]
    PID = pdict[process_id]

    mc_writedir = PID['PARAMS']['motion']['destdir']
    
    if metric == 'corr_mean':
        
