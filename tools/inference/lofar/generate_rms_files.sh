#!/bin/bash  

source /home/blao/RGC-Mask-Transfiner/bashrc
module use /home/app/modulefiles
module load mpich/cpu-3.2.1-gcc-4.8.5

srun --mpi=pmi2 -p hw -N 1 -n 2 --ntasks-per-node 2 python generate_rms_files.py --inpdir /p9550/LOFAR/LoTSS-DR1/Mingo_fits --outdir /p9550/LOFAR/LoTSS-DR1/Mingo_rms --fitslists /p9550/LOFAR/LoTSS-DR1/mingo_fits.txt

