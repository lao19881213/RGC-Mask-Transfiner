#!/bin/bash


IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"


for fn in `cat $IMG_FILES`;
do

    FIST_DIR="${DATA_DIR}/RACS-DR1_${fn}_split_fits_png"
    cd $FIST_DIR    
    LIST_FILE="${DATA_DIR}/RACS-DR1_${fn}.txt"
    ls *.fits > $LIST_FILE
    LIST_FILE="${DATA_DIR}/RACS-DR1_${fn}_png.txt"
    ls *.png > $LIST_FILE

done
