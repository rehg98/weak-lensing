#!/bin/bash
#SBATCH -N 1
#SBATCH -n 28
#SBATCH --ntasks-per-node=28
#SBATCH -t 2:00:00
#SBATCH --output=galgaussSNRout.out
#SBATCH --error=galgaussSNRerr.err
#SBATCH --mail-type=all
#SBATCH --mail-user=rgolant@princeton.edu
module load openmpi
module load intel
module load anaconda3/4.3.0
srun -n 28 python galgauss_SNR_norm.py
wait
