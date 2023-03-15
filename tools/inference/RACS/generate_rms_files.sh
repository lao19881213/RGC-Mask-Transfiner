#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"


for fn in `cat $IMG_FILES`;
do
    RMS_DIR=$DATA_DIR/RACS-DR1_${fn}_split_rms
    if [ ! -d "${RMS_DIR}" ];then
       mkdir -p ${RMS_DIR}
    else
       echo "${RMS_DIR} is already exists"
    fi
    echo "Processing ${RMS_DIR} ... ..."

    mpirun -np 1 python generate_rms_files.py \
        --inpdir $DATA_DIR/RACS-DR1_${fn}_split_fits_png \
        --outdir $RMS_DIR \
        --fitslists $DATA_DIR/RACS-DR1_${fn}.txt

done

