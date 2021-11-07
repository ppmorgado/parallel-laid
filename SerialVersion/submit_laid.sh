#!/bin/bash

#SBATCH --job-name=LaidS                  # submit_check.sh
#SBATCH --time=48:01:01                   # max time
#SBATCH --ntasks=1                        # total number of tasks
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
mpirun python laid_serial_6.py $SLURM_ARRAY_TASK_ID

echo "Finished job $SLURM_JOBID"
echo $(date +"%F %T,%N")

squeue -n pmorgado

echo "disk usage after"
du -hs
# echo "h5dump:"
# h5dump -H -p -d dmatrix laidp_dmatrix_0.h5
# echo "h5ls:"
# h5ls -lrv laidp_dmatrix_0.h5
# echo "End hdf5 file"
rm laidp_dmatrix_0.h5
