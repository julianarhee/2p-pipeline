
import os
import json
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

class StimInfo:
    def _init_(self):
        self.stimid = ''
        self.trials = []
        self.frames = []
        self.frames_sec = []

def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__

source = '/nas/volume1/2photon/projects'
experiment = 'scenes'
session = '20171003_JW016'
acquisition = 'FOV1'
functional_dir = 'functional'


acquisition_dir = os.path.join(source, experiment, session, acquisition)

path_to_functional = os.path.join(acquisition_dir, functional_dir)
paradigm_dir = 'paradigm_files'
path_to_paradigm_files = os.path.join(path_to_functional, paradigm_dir)

stiminfo_basename = 'stiminfo'

# Load reference info:
ref_json = 'reference_%s.json' % functional_dir 
with open(os.path.join(acquisition_dir, ref_json), 'r') as fr:
    ref = json.load(fr)

# Load SI meta data:
si_basepath = ref['raw_simeta_path'][0:-4]
simeta_json_path = '%s.json' % si_basepath
with open(simeta_json_path, 'r') as fs:
    simeta = json.load(fs)

nframes = int(simeta['File001']['SI']['hFastZ']['numVolumes'])
framerate = float(simeta['File001']['SI']['hRoiManager']['scanFrameRate'])
volumerate = float(simeta['File001']['SI']['hRoiManager']['scanVolumeRate'])
frames_tsecs = np.arange(0, nframes)*(1/volumerate)


# frame info:
first_frame_on = 50
stim_on_sec = 0.5
iti = 2.
vols_per_trial = 15

nframes_on = stim_on_sec * volumerate
nframes_off = vols_per_trial - nframes_on
frames_iti = round(iti * volumerate)

# Create stimulus-dict:
stimdict = dict()
for fi in range(nfiles):
    currfile = "File%03d" % int(fi+1)
    stim_fn = 'stim_order.txt'

    # Load stim-order:
    with open(os.path.join(path_to_paradigm_files, stim_fn)) as f:
        stimorder = f.readlines()
    curr_stimorder = [l.strip() for l in stimorder]
    unique_stims = sorted(set(curr_stimorder), key=natural_keys)
    first_frame_on = 50
    for trialnum,stim in enumerate(curr_stimorder):
        print "Stim on frame:", first_frame_on
        if not stim in stimdict.keys():
            stimdict[stim] = dict()
        if not currfile in stimdict[stim].keys():
            stimdict[stim][currfile] = StimInfo() 
            stimdict[stim][currfile].trials = []
            stimdict[stim][currfile].frames = []
            stimdict[stim][currfile].frames_sec = []

        framenums = list(np.arange(int(first_frame_on-frames_iti), int(first_frame_on+(vols_per_trial))))
        frametimes = [frames_tsecs[f] for f in framenums]
        stimdict[stim][currfile].trials.append(trialnum)       
        stimdict[stim][currfile].frames.append(framenums)
        stimdict[stim][currfile].frames_sec.append(frametimes)
        first_frame_on = first_frame_on + vols_per_trial


stiminfo_json = '%s.json' % stiminfo_basename
stiminfo_mat = '%s.mat' % stiminfo_basename

with open(os.path.join(path_to_paradigm_files, stiminfo_json), 'w') as fw:
    json.dumps(jsonify(stimdict), fw, sort_keys=True, indent=4)
scipy.io.savemat(os.path.join(path_to_paradigm_files, stiminfo_mat), mdict=stimdict)
