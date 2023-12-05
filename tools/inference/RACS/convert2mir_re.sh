#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part1_fits_fixed_re.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"


for fn in `cat $IMG_FILES`;
do
    SBID=${fn: 21:7}
    FIST_DIR="${DATA_DIR}/${SBID}_split_fits_png"
    
    LIST_FILE="${DATA_DIR}/${SBID}.txt"
    MIR_DIR="${SBID}_split_mir"
    
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
