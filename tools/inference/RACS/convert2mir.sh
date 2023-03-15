#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"


for fn in `cat $IMG_FILES`;
do
    FIST_DIR="${DATA_DIR}/RACS-DR1_${fn}_split_fits_png"
    
    LIST_FILE="${DATA_DIR}/RACS-DR1_${fn}.txt"
    MIR_DIR="RACS-DR1_${fn}_split_mir"
    
    if [ ! -d "${DATA_DIR}/${MIR_DIR}" ];then
       mkdir -p ${DATA_DIR}/${MIR_DIR}
    else
       echo "${DATA_DIR}/${MIR_DIR} is already exists"
    fi
    echo "Processing ${MIR_DIR} ... ..."
    
    python convert2mir.py --fitsdir ${FIST_DIR} \
                          --fitslist ${LIST_FILE} \
                          --mirdir ${DATA_DIR}/${MIR_DIR}

done
