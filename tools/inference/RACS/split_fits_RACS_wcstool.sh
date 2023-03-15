#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt" 

IMG_DIR=/home/data0/lbq/inference_data/ASKAP_RACS

for fn in `cat $IMG_FILES`;
do
    if [ ! -d "${IMG_DIR}/RACS-DR1_${fn}_split_fits_png" ];then
       mkdir -p ${IMG_DIR}/RACS-DR1_${fn}_split_fits_png
    else
       echo "${IMG_DIR}/RACS-DR1_${fn}_split_fits is already exists"
    fi
    echo "Processing ${fn} ... ..."
    python split_fits_wcstool.py --fname ${IMG_DIR}/image_cubes/RACS-DR1_${fn}.fits --workdir ${IMG_DIR}/RACS-DR1_${fn}_split_fits_png --ImageSize 300 --OverlappingSize 40 #10 arcmin
    echo "Split ${fn} image is done."
done

