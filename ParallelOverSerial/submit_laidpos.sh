#!/bin/bash

#SBATCH --job-name=LaidPoS                # submit_check.sh
#SBATCH --time=24:24:24                   # max time
#SBATCH --ntasks=5                        # Number of MPI ranks  (total number of tasks)
#SBATCH --nodes=1                         # Run all processes on a x nodes
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

# pwd; hostname; date
# echo "Start job: $SLURM_JOBID"
# echo $(date +"%F %T,%N")

# clean and load environment
module purge
module load hdf5/1.12.0

module list
echo "python version"
python --version

rm laidp_aux.h5

# job payload
mpirun python laid_serial_6b.py

echo ""
echo "JobId finished                 = $SLURM_JOBID"
echo "Date End                       = $(date +"%F %T,%N")"
