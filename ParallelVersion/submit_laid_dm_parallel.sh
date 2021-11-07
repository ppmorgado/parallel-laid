#!/bin/bash

#SBATCH --job-name=LaidDMPar              # submit_check.sh
#SBATCH --time=48:01:01                   # max time
#SBATCH --ntasks=5                        # total number of tasks
#SBATCH --partition=hpc                   # CIRRUS-A (Lisbon)

echo "JobId                          = $SLURM_JOBID"
echo "Date/Time Start                = $(date +"%F %T,%N")"
echo "Hostname                       = $(hostname -s)"
echo "Working Directory              = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "SLURM_ARRAY_TASK_ID            = $SLURM_ARRAY_TASK_ID"
echo "Running LAID parallel with serial code on $SLURM_CPUS_ON_NODE CPU cores"


# clean and load environment
module purge
module load hdf5/1.12.0

# display environment
echo "module loaded"
module list
echo "python version"
python --version
echo "working dir"
pwd
echo "disk usage before"
du -hs

# job payload

# T6
mpirun python laid_parallel_dm_2.py $SLURM_ARRAY_TASK_ID

echo ""
echo "JobId finished                 = $SLURM_JOBID"
echo "Date End                       = $(date +"%F %T,%N")"

# squeue -n pmorgado

echo "disk usage after"
du -hs
echo "h5dump:"
h5dump -H -p -d dmatrix laidp_dmatrix_0.h5
echo "h5ls:"
h5ls -lrv laidp_dmatrix_0.h5
echo "End hdf5 file"
# rm laidp_dmatrix_.h5
