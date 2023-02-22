#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results
DATA_DIR=/home/data0/lbq/inference_data
CODE_DIR=/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST


ranks="0 1 2 3 4 5 6 7"

for rank in $ranks;
do
echo "Processing cj_${rank} ... ..."

python ${CODE_DIR}/find_duplicate_extended_cj.py \
     --fr12csv ${RESULT_DIR}/FIRST_infer_part0-4_cj.csv \
     --inpdir ${DATA_DIR}/FIRST_fits \
     --outdir ${RESULT_DIR} \
     --outfile FIRST_infer_part0-4_cj_rep_${rank}.txt \
     --rank ${rank}
done


