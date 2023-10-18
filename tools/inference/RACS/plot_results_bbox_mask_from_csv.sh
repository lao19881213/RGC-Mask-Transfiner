#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part1_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"


for fn in `cat $IMG_FILES`;
do
    SBID=${fn: 21:7}
    PRED_DIR="${SBID}_pred"

    if [ ! -d "${DATA_DIR}/${PRED_DIR}" ];then
       mkdir -p ${DATA_DIR}/${PRED_DIR}
    else
       echo "${DATA_DIR}/${PRED_DIR} is already exists"
    fi
    echo "Processing ${PRED_DIR} ... ..."

    python plot_results_bbox_mask_from_csv.py \
         --resultsfile /home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-mid_${SBID}.csv \
         --imagedir $DATA_DIR/${SBID}_split_fits_png \
         --outdir $DATA_DIR/${SBID}_pred 

done
