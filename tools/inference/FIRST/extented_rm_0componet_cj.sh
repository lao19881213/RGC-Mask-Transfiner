#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results

python extented_rm_0componet.py \
     --incsv ${RESULT_DIR}/FIRST_infer_part0-4_cj_final.csv \
     --compcsv ${RESULT_DIR}/extended_components_cj.csv



