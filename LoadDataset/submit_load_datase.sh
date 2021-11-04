#!/bin/bash

#SBATCH --job-name=LaidpM1                # submit_check.sh
#SBATCH --time=7:10:10                    # max time
#SBATCH --ntasks=2                        # total number of tasks
#SBATCH --partition=hpc                   # CIRRUS-A (Lisbon)

echo "Start job: $SLURM_JOBID"
echo $(date +"%F %T,%N")

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

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
mpirun python laidp_load_dataset.py $SLURM_ARRAY_TASK_ID

echo "Finished job $SLURM_JOBID"
echo $(date +"%F %T,%N")

echo "disk usage after"
du -hs
echo "h5dump:"
h5dump -H -A -p -d database laidp_original_dataset.h5
echo "h5ls:"
h5ls -lrv laidp_original_dataset.h5
echo "End hdf5 file"
