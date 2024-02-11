#!/bin/bash


IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part2_fits_fixed.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"


for fn in `cat $IMG_FILES`;
do
    SBID=${fn: 21:7}
    FIST_DIR="${DATA_DIR}/${SBID}_split_fits_png"
    cd $FIST_DIR    
    LIST_FILE="${DATA_DIR}/${SBID}.txt"
    ls *.fits > $LIST_FILE
    LIST_FILE="${DATA_DIR}/${SBID}_png.txt"
    ls *.png > $LIST_FILE

done
