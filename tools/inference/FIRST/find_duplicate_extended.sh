#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results
DATA_DIR=/home/data0/lbq/inference_data
CODE_DIR=/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST

classes="fr2 ht cj"

for cls in $classes;
do

echo "Processing ${cls} ... ..."

python ${CODE_DIR}/find_duplicate_extended.py \
     --fr12csv ${RESULT_DIR}/FIRST_infer_part0-4_${cls}.csv \
     --inpdir ${DATA_DIR}/FIRST_fits \
     --outdir ${RESULT_DIR} \
     --outfile FIRST_infer_part0-4_${cls}_rep.txt

done



