#!/bin/bash 
#SBATCH -N 1 # node count 
#SBATCH -n 28
#SBATCH --ntasks-per-node=28 
#SBATCH -t 1:00:00 
#SBATCH --output=SNRout.out
#SBATCH --error=SNRerr.err
#SBATCH --mail-type=all
#SBATCH --mail-user=rgolant@princeton.edu 

module load openmpi
module load intel


srun -n 28 python parSNR.py 
wait
