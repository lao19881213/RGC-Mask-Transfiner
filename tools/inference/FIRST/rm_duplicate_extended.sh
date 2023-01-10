#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results
DATA_DIR=/home/data0/lbq/inference_data
CODE_DIR=/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST

classes="fr1 fr2 ht cj"

for cls in $classes;
do

echo "Processing ${cls} ... ..."


python ${CODE_DIR}/rm_duplicate_extended.py \
     --fr12csv ${RESULT_DIR}/FIRST_infer_part0-4_${cls}.csv \
     --reptxt ${RESULT_DIR}/FIRST_infer_part0-4_${cls}_rep.txt \
     --outdir ${RESULT_DIR} \
     --outcsv FIRST_infer_part0-4_${cls}_final.csv

done



