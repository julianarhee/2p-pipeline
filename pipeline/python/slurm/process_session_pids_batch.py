#!/bin/bash
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-00:10 # Runtime in D-HH:MM
#SBATCH -p cox # Partition to submit to
#SBATCH --mem=8192 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be writtenhostname

module load matlab/R2015b-fasrc01
module load python/2.7.13-fasrc01

source activate 2pdev

python /n/coxfs01/2p-pipeline/repos/2p-pipeline/python/process_pids_for_session.py -R/n/coxfs01/julianarhee/testdata -iJR063 -S20171128_JR063_slurmtest --slurm

source deactivate

