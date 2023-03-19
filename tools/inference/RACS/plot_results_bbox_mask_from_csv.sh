#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"


for fn in `cat $IMG_FILES`;
do
    PRED_DIR="RACS-DR1_${fn}_pred"

    if [ ! -d "${DATA_DIR}/${PRED_DIR}" ];then
       mkdir -p ${DATA_DIR}/${PRED_DIR}
    else
       echo "${DATA_DIR}/${PRED_DIR} is already exists"
    fi
    echo "Processing ${PRED_DIR} ... ..."

    python plot_results_bbox_mask_from_csv.py \
         --resultsfile /home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-DR1_${fn}.csv \
         --imagedir $DATA_DIR/RACS-DR1_${fn}_split_fits_png \
         --outdir $DATA_DIR/RACS-DR1_${fn}_pred 

done
