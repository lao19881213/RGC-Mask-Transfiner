#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


mpirun -np 1 python generate_rms_files.py \
        --inpdir /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_split_fits_png \
        --outdir /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_split_rms \
        --fitslists /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A.txt

