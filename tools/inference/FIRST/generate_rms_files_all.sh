#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

mpirun -np 1 python generate_rms_files.py --inpdir /home/data0/lbq/inference_data/FIRST_fits --outdir /home/data0/lbq/inference_data/FIRST_rms --fitslists /home/data0/lbq/inference_data/FIRST.txt

