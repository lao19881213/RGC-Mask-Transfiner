#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part1_fits_fixed.txt" 

IMG_DIR=/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid

PART='part1_conv'

for fn in `cat $IMG_FILES`;
do
    SBID=${fn: 21:7}
    echo $SBID

    if [ ! -d "${IMG_DIR}/${SBID}_split_fits_png" ];then
       mkdir -p ${IMG_DIR}/${SBID}_split_fits_png
    else
       echo "${IMG_DIR}/${SBID}_split_fits_png is already exists"
    fi
    echo "Processing ${fn} ... ..."
    python split_fits_RACS-mid.py --fname ${IMG_DIR}/${PART}/${fn} --workdir ${IMG_DIR}/${SBID}_split_fits_png --ImageSize 300 --OverlappingSize 40 #10 arcmin
    echo "Split ${fn} image is done."
done

