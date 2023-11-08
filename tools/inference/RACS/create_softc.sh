#!/bin/bash

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part1_fits_fixed.txt"

IMG_DIR=/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid

PART='part1'

for fn in `cat $IMG_FILES`;
do
    SBID=${fn: 21:7}
    echo $SBID
    cd /home/data0/lbq/RGC-Mask-Transfiner   
    ln -s ${IMG_DIR}/${SBID}_split_mir ${SBID}_split_mir

done
