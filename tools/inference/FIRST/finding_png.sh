#!/bin/bash

#SBATCH --nodes=10
#SBATCH --job-name=plot
##SBATCH --time=02:00:00
#SBATCH --error=err-%j.out
#SBATCH --partition=hw
#SBATCH -n 300
#SBATCH -x hw-x86-cpu09

source /home/blao/RGC-Mask-Transfiner/bashrc
module use /home/app/modulefiles
module load mpich/cpu-3.2.1-gcc-4.8.5

srun --mpi=pmi2  python /home/blao/RGC-Mask-Transfiner/tools/inference/FIRST/finding_png.py 

