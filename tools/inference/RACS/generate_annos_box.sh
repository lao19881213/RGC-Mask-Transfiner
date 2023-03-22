#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"
DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"
RESULT_DIR="/home/data0/lbq/RGC-Mask-Transfiner/RACS_results"


for fn in `cat $IMG_FILES`;
do
    ANNO_DIR="RACS-DR1_${fn}_anno"
    
    if [ ! -d "${DATA_DIR}/${ANNO_DIR}" ];then
       mkdir -p ${DATA_DIR}/${ANNO_DIR}
    else
       echo "${DATA_DIR}/${ANNO_DIR} is already exists"
    fi
    echo "Processing ${ANNO_DIR} ... ..."

    cp ${DATA_DIR}/RACS-DR1_${fn}_split_fits_png/*.png ${DATA_DIR}/${ANNO_DIR}/
    
    python generate_annos_box.py \
           --result ${RESULT_DIR}/RACS-DR1_${fn}.csv \
           --inpdir ${DATA_DIR}/${ANNO_DIR} \
           --outdir ${DATA_DIR}/${ANNO_DIR}

done
