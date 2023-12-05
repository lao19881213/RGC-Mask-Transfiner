#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part1_fits_fixed_re.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"


for fn in `cat $IMG_FILES`;
do
    SBID=${fn: 21:7}
    RMS_DIR=$DATA_DIR/${SBID}_split_rms
    if [ ! -d "${RMS_DIR}" ];then
       mkdir -p ${RMS_DIR}
    else
       echo "${RMS_DIR} is already exists"
    fi
    echo "Processing ${RMS_DIR} ... ..."

    python generate_rms_files.py \
        --inpdir $DATA_DIR/${SBID}_split_fits_png \
        --outdir $RMS_DIR 
done

