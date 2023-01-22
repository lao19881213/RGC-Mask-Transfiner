#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results

python /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/FIRST_fits_download.py \
       --catalogfile ${RESULT_DIR}/nomatch_hetu_final.csv \
       --outdir /home/data0/lbq/inference_data/nomatch_cs_hetu 
