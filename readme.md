## Description

Temporary set of scripts for imaging pipeline that combines visual stimulation and imaging datasets, extracts calcium traces (and ROIs), and parses data into trials. GUI option is for quick and simple visualization of 2D FOVs. 

Written by Juliana Rhee (Cox Lab).

## Sources

Parts of the pipeline use one or more of the following Github repos: 
Acquisition2P_class (Harvey Lab, Harvard), helperFunctions (Harvey Lab, Harvard), NoRMCorre (Simons Foundation), ca_source_extraction (Eftychios Pnevmatikakis), CaImAn (Simons Foundation) 

## Code Example

Basic workflow currently uses both Python and Matlab, until we choose a smaller subset of methods to use.
```
# 1. Pre-processing

python ./python/process_raw.py # Get metadata from raw TIFFs, remove extra flyback frames, if needed. 

matlab scripts:

init_header;                     # Must be edited. User-specified inputs to parameters for preprocessing
check_init;                      # Check specified inputs from init_header.m 
initialize_analysis;             # Create data structures for running analysis with specified params 
process_tiffs(I, A, new_mc_id);  # Process tiffs with params specified in struct I, acquisition info in struct A

# 2. ROI extraction

python roi_blob_detector.py      # Segment blobs using projected times series of a given slice (currently, average)


# 3. Trace extraction

matlab scripts:

init_header;                    # Must be edited. Same as above.
check_init;                     # Same as above. 
initialize_analysis;            # Same as above.
get_traces_from_rois(I, A);     # Must have ROI masks and info stored in '<acquisition_dir>/ROIs'. 


# 4. Trial alignment

python /paradigm/parse_mw_trials.py                 # Extract trial info from behavioral paradigm (.mwk)
python /paradigm/extract_acquisition_events.py      # Format trial structure in a standardized way
python /paradigm/create_stimdict.py                 # Parse trial info by stimulus, assign frames and frame times
python /paradigm/files_to_trials.py                 # Combine stimulus and trial info from acquistiion across TIFFs
python /paradigm/save_roi_dicts.py                  # Calculate df/f (and other things) for each ROI


# 5. Visualization

python /paradigm/roi_subplots_by_stim.py            # For each ROI, create PSTH-style plots of each stimulus 
python /paradigm/plot_slice_rois.py                 # Plot each ROI on average slice image.

```
Notes: Python scripts require additional options specific for the experiment (add -h handle to view arguments and descriptions). See demo_pipeline.py (.mat) for more info. 

## Motivation

Potential resource to start a semi-automated pipeline for quick viewing of datasets. Make modular enough for added features from other users.

## Installation

Developed with Python2 and Matlab R2015b. Python environments can be cloned with Anaconda using the environment.yml file, which will create an Anaconda env called 2pdev, or create a new env and install reqs:
```
pip install -r requirements.txt
```


