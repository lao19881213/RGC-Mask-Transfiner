#!/bin/bash

source /home/blao/RGC-Mask-Transfiner/bashrc
module use /home/app/modulefiles
module load mpich/cpu-3.2.1-gcc-4.8.5

export http_proxy=http://192.168.6.12:3128
export https_proxy=https://192.168.6.12:3128

srun --mpi=pmi2 -p hw -N 1 -n 1 --ntasks-per-node 1 python /home/blao/RGC-Mask-Transfiner/tools/inference/FIRST/J_ApJS_HT_download.py \
     --catalogfile  J_ApJS_259_31_table1.dat.csv \
     --outdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT
