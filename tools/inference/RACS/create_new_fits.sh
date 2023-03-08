#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

IMG_FILES="0037-06A" 

IMG_DIR=/home/data0/lbq/inference_data/ASKAP_RACS

for fn in $IMG_FILES;
do
python create_new_fits.py \
       --inpdir ${IMG_DIR}/RACS-DR1_${fn}_split_fits \
       --outdir ${IMG_DIR}/RACS-DR1_${fn}_split_fits
done
