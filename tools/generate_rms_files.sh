#!/bin/bash  

source /home/blao/rgz_resnet_fpn/bashrc
module use /home/app/modulefiles
module load mpich/cpu-3.2.1-gcc-4.8.5

srun --mpi=pmi2 -p hw -N 1 -n 2 --ntasks-per-node 2 python generate_rms_files.py --inpdir /home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR --outdir /home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR_rms --fitslists /home/blao/RGC-Mask-Transfiner/datasets/coco/lofar_fits.txt

