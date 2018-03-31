#!/bin/bash
# collate_coreg_results.sbatch
#
#SBATCH -J coreg-collate # A single job name for the array
#SBATCH -p cox # run on cox gpu to use correct env
#SBATCH -n 12 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-2:00 # Running time of 1 hours
#SBATCH --mem 4000 # Memory request of 4 GB
#SBATCH -o log/coreg_collate_%A_f%a.out # Standard output
#SBATCH -e log/coreg_collate_%A_f%a.err # Standard error

# load modules
module load matlab/R2015b-fasrc01
module load Anaconda/5.0.1-fasrc01

# activate 2p-pipeline environment:
source activate /n/coxfs01/2p-pipeline/envs/pipeline

# grab filename from array exported from 'parent' shell:
PARAMSPATH=${1}
echo "COREG COLLATE p info: ${PARAMSPATH}"

# Collate coregsitration output
python /n/coxfs01/2p-pipeline/repos/2p-pipeline/pipeline/python/slurm/rois/collate_coreg_results.py $PARAMSPATH

