#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results
DATA_DIR=/home/data0/lbq/inference_data
CODE_DIR=/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST



python ${CODE_DIR}/cross_match_ht.py \
     --ApJscsv ${RESULT_DIR}/J_ApJS_HT_catalog.csv \
     --result ${RESULT_DIR}/FIRST_infer_part0-4_ht_final.csv \
     --inpdir ${DATA_DIR}/FIRST_fits \
     --outdir ${RESULT_DIR} 





