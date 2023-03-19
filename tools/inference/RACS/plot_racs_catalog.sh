#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"


for fn in `cat $IMG_FILES`;
do

    if [ ! -d "${DATA_DIR}/RACS-DR1_${fn}_split_racs_pred" ];then
       mkdir -p ${DATA_DIR}/RACS-DR1_${fn}_split_racs_pred
    else
       echo "${DATA_DIR}/RACS-DR1_${fn}_split_racs_pred is already exists"
    fi
    echo "Processing ${DATA_DIR}/RACS-DR1_${fn}_split_racs_pred ... ..."

    python plot_racs_catalog.py --racscsv ${DATA_DIR}/catalogue/AS110_Derived_Catalogue_racs_dr1_gaussians_galacticcut_v2021_08_v02_5723.csv \
                              --inpdir ${DATA_DIR}/RACS-DR1_${fn}_split_fits_png \
                              --outdir ${DATA_DIR}/RACS-DR1_${fn}_split_racs_pred

done 
