#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

IMG_FILES="SB45291" 

IMG_DIR=/home/data0/lbq/inference_data/ASKAP_RACS

for fn in $IMG_FILES;
do
    if [ ! -d "${IMG_DIR}/${fn}_split_fits" ];then
       mkdir -p ${IMG_DIR}/${fn}_split_fits
    else
       echo "${IMG_DIR}/${fn}_split_fits is already exists"
    fi
    echo "Processing ${fn} ... ..."
    python split_fits_new.py --fname ${IMG_DIR}/image_cubes/${fn}.fits --workdir ${IMG_DIR}/${fn}_split_fits --ImageSize 250 --OverlappingSize 40 #10 arcmin
    echo "Split ${fn} image is done."
done

#w: 15283/132 
#h: 13437/132
