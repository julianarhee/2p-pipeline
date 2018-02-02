# Runnine PIPELINE on the cluster with SLURM
Several portions of the pipeline can be run much faster and in parallel if using the cluster. For example, running multiple parameter sets of the same .tif sources, extracting ROIs from individual .tif files, etc. These are a few example scripts to get started.

### TIFF processing.
1 As always, create the parameter set for all runs to be processed:
```
python set_pid_params.py --notnative -H /nas/volume1/2photon/data -D /n/coxfs01/2p-data -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -s raw -t raw --bidi --motion -f 5 --default --slurm
```
Here, we specifiy --notnative if not creating the param set on Odyssey. This is useful if still rsync-ing data over, etc. If data is already living at ROOTDIR, and parameters are being created at ROOTDIR, as well, there is no need for the first three parameters in the input example above. In this example, we want to run bidi-correction and motion-correction on the raw data, using File005 as the reference, and default params otherwise.

If there is more than one PID set to process for this session or run, create a tmp file that tracks all PIDs to run:
```
python create_session_pids.py -i <ANIMALID>  -S <SESSION> --slurm --indie
``
Set --notnative, -D, and -H, if relevant. This creates a set of tmp files in `SESSIONDIR/tmp_spids/.json` for each PID set in the session.

2. After logging into RC, and creating a job output dir and run the shell script to process all PIDs associated with the session:
```
bash /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/batch_process_pids.sh <ANIMALID> <SESSION> [<ACQUISITION> <RUN> <PIDHASH>]
```
Only the first two args are required if you are running a set of session PIDs. If you only want to run a single PID set (and you did not run `create_session_pids.py`), add the last three args in the specified order. 

### ROI extraction.
#### CaImAn (caiman2D)
1. Create the ROI param set:
```
python set_roi_params.py -i <ANIMALID> -S <SESSION> -A <ACQUISITION> -R <RUN> -o caiman2D
```
2. Login to RC and run the shell script to extract cNMF ROIs.
```
ssh RCUSERNAME@login.rc.fas.harvard.edu  # Login to RC
mkdir ~/slurmjobs/roiset001 		 # (optional) tmp dir for slurm output
cd ~/slurmjobs/roiset001

bash /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/nmf_tiff_array.sh <ANIMALID> <SESSION> <ROIDHASH> <LASTFILENUM> <FIRSTFILENUM>
```
The last two args are optional. If LASTFILENUM is not provided, default value is 1 (same for FIRSTFILENUM). This allows you to extract ROIs from File001.tif through File006.tif, or File002.tif through File005.tif.

This creates a .out and .err file for each processed file. The main python script called by the sbatch script will also create log files in /tmp_rids/logging/ dir.


