## Description

Temporary set of scripts for imaging pipeline that combines visual stimulation and imaging datasets, extracts calcium traces (and ROIs), and parses data into trials. GUI option is for quick and simple visualization of 2D FOVs. 

Written by Juliana Rhee (Cox Lab).

## Sources

Parts of the pipeline use one or more of the following Github repos: 
Acquisition2P_class (Harvey Lab, Harvard), helperFunctions (Harvey Lab, Harvard), NoRMCorre (Simons Foundation), ca_source_extraction (Eftychios Pnevmatikakis), CaImAn (Simons Foundation) 

## Code Example

Basic workflow currently uses both Python and Matlab, until we choose a smaller subset of methods to use.
```
python create_substacks.py 
matlab -r 'preprocess';
python parse_mw_trials.py 
matlab -r 'create_acquisition_struct';
```
Notes: python scripts require additional options specific for the experiment (add -h handle to view arguments and descriptions). edit the above matlab scripts to input experiment-specific parameters before running.

## Motivation

Potential resource to start a semi-automated pipeline for quick viewing of datasets. 

## Installation

Developed with Matlab R2015b. Python environments can be cloned with Anaconda using the environment.yml file.


