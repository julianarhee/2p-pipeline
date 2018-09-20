# 2p-pipeline
Set of scripts for imaging pipeline that combines visual stimulation and imaging datasets, extracts calcium traces (and ROIs), and parses data into trials. GUI option is for quick and simple visualization of 2D FOVs. 

Written by Juliana Rhee (Cox Lab).

## Motivation

Potential resource to start a semi-automated pipeline for quick viewing of datasets. Make modular enough for added features from other users.

## Sources
Parts of the pipeline use one or more of the following Github repos: 
Acquisition2P_class (Harvey Lab, Harvard), helperFunctions (Harvey Lab, Harvard), NoRMCorre (Simons Foundation), ca_source_extraction (Eftychios Pnevmatikakis), CaImAn (Simons Foundation) 

## Getting Started
Create the pipeline environment (conda).
```
$ conda env create -f environment.yml -p /path/to/conda/envs/pipeline
$ source activate pipeline
$ cd ./2p-pipeline
$ python setup.py --install

Install the matlab engine for python:
$ cd /path/to/matlab-R2015b/extern/engines/python
$ python setup.py build --build-base='/tmp' install --prefix='/path/to/conda/envs/pipeline'

Also need to install the CaImAn package to pipeline env (see github for install details).
```
NOTE:  The pipeline assumes a standard file-tree:
```
/ROOTDIR/                                       # rootdir of all data [/nas/volume1/2photon/data]
└── ANIMALID                                    # animal name [JR016]
    └── SESSION                                 # session id with format YYYYMMDD [20170130]
        └── ACQUISITION01                       # FOV, imaging params, etc. [FOV1_zoom1x_volume] 
            └── RUN001                          # experiment run or protocol. [gratings_static_run1] 
                └── raw                         # raw .tif files acquired (SI 2016)
                    ├── file_00001.tif
                    ├── file_00002.tif
                    ├── file_00003.tif
                    ├── file_00004.tif
                    └── paradigm_files          # raw behavior/stimulus-related files
                        ├── protocol_output.mwk
                        └── serial_data.txt
```

## Code Example
Basic workflow currently uses both Python and Matlab, until we choose a smaller subset of methods to use.

### 1. Pre-processing
a. Create a set of processing params (PID set): 
```
python pipeline/python/set_pid_params.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> --bidi --motion -f 2 --default
```
Any processing step writes to `<ROOTDIR>/<ANIMALID>/<SESSION>/<ACQUISITION>/<RUN>/processed`.  Temprory parameter sets are stored in `.../processed/tmp_pids/tmp_pid_<pidhash>.json`, which contains all relevant fields for running preprocessing steps. Here, we want *bidi*rectional scan correction and *motion*-correction, using *File002* as reference, and *default* params for all else.

A unique 6-char hash is created for the specified parameter set. The tmp file will move to a 'completed' subdir if the process exits without error.  See pipeline/python/process_session.txt for details.

b.  Process the raw .tif files:
```
python pipeline/python/preprocessing/process_raw.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -p <PIDHASH>
```
Processed .tif files are saved within the process ID folder, and output .tifs of each step are saved in separate subfolders. For example: `.../processed/processed001_<PIDHASH>/bidi`, `.../processed/processed001_<PIDHASH>/mcorrected`, etc. All output dirs are hashed and folder names are updated with the directory hash.

### 2. ROI extraction
a.  Create a set of ROI extraction params (RID set):
```
python pipeline/python/set_roi_params.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -s processed -t  mcorrected -o caiman2D -g mean 
```
All ROI sets are saved to `<ROOTDIR>/<ANIMALID>/<SESSION>/ROIs`.  Tmp ROI param files are stored in `.../ROIs/tmp_rids/tmp_rid_<RIDHASH>.json`, and moved to `completed` subdir if ROIs are extracted without error. Here, we want to extract *CaImAn 2D* ROIs from the .tif files found in the *motion-corrected* dir of the specified run, and use the *mean* slice image for visualization.

b.  Extract ROIs:
```
python pipeline/python/get_rois.py -i <ANIMALID> -S <SESSION> -r <ROI_ID> 
```
Since different ROI extraction methods will produce different output formats, this script additionally formats ROIs to a rough standard, and saves the standard output .hdf5 file and a set of .png files that show the masks drawn on a z-projected image. For example: `.../ROIs/rois001_<RIDHASH>/masks.hdf5`, `.../ROIs/rois001_<RIDHASH>/figures/rois_File00X_masks.png`, etc. Additionally, roiparams.json is saved in the ROI set dir.

### 3. Trace extraction
a.  Create a set of trace extraction params (TRACE set):
```
python pipeline/python/set_trace_params.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -s processed001 -t mcorrected -o rois001 -c 1
```
All trace sets are saved to `<ROOTDIR>/<ANIMALID>/<SESSION>/<ACQUISITION>/<RUN>/traces`. Tmp trace params are saved to `.../traces/tmp_tids/tmp_tid_<TRACEHASH>.json`, and moved to `completed` when successfully completed. Here, we want to extract traces using the roi set *rois001* applied to *Channel01* of motion-correction .tif files from processing-set *processed001*.

b.  Extract traces:
```
python pipeline/python/traces/get_traces.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -t traces001
```
Extracted traces and meta info are saved per .tif file in `.../traces/traces001_<TRACEHASH>/files/*.hdf5`, ROIs used in extraction are saved to `.../traces/traces001_<TRACEHASH>/figures/masks/*.png`. Each .tif is kept separate (saved to `.../traces/traces001_<TRACEHASH>/files/*.hdf5`) at this stage. Masks are applied to raw .tif stacks, and preprocessing of ROI traces is done in the next step. Neuropil correction is also done at this step (Use -h to get neuropil options).

### 4. Trial alignment
a.  Parse acquisition events into trials:
```
python ./paradigm/extract_acquisition_events.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN>
```
Parsed trial and frame info is saved to `<ROOTDIR>/<ANIMALID>/<SESSION>/<ACQUISITION>/<RUN>/paradigm`.  Default stimulus- and behavior-acquisition method uses MWORKS, combined with serial data collected at 1kHz saved to a .txt file. These acquisition files should be read-only in the `paradigm_files` subdir of the `raw` data folder that contains the corresponding .tif files.

b.  Align timecourses to acquisition events:
```
python ./paradigm/tifs_to_data_arrays.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -t traces001 
```

All .tif files in a block (or set of blocks) are parsed and aigned to trials for each ROI. Use -h to see preprocessing options (e.g., smoothing, drift-correction) and trial alignment options (e.g., pre-ITI baseline duration, post-stimulus offset duration).

### 5. Visualization
```
python /paradigm/plot_responses.py                  # For each ROI, create PSTH-style plots of each stimulus (Use -h to see options for arranging figure grid based on stimulus config).
python /visualization/plot_session_summary.py       # Plot session summary based on stimulus (i.e., experiment) type. *NOTE: Runs are combined at this step, if specified. Stats to determine visual/selective cells.
python /visualization/plot_roi_timecourses.py       # Plot timecourse for a specified ROI for all files. High-light stim reps.
```

Notes: Python scripts require additional options specific for the experiment (add -h handle to view arguments and descriptions). See demo_pipeline.py (.mat) for more info. 


