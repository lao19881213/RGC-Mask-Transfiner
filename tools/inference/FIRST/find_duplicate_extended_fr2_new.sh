#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results
DATA_DIR=/home/data0/lbq/inference_data
CODE_DIR=/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST


echo "Processing fr2 ... ..."

python ${CODE_DIR}/find_duplicate_extended_fr2_new.py \
     --fr12csv ${RESULT_DIR}/FIRST_infer_part0-4_fr2.csv \
     --inpdir ${DATA_DIR}/FIRST_fits \
     --outdir ${RESULT_DIR} \
     --outfile FIRST_infer_part0-4_fr2_rep.txt




