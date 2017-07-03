#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 28
#SBATCH --ntasks-per-node=28
#SBATCH -t 1:00:00
#SBATCH --output=SNRout.out
#SBATCH --error=SNRerr.err
#SBATCH --mail-type=all
#SBATCH --mail-user=rgolant@princeton.edu
module load openmpi
module load intel
module load anaconda3/4.3.0
srun -n 28 python parSNR.py

