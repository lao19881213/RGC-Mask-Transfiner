#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results
DATA_DIR=/home/data0/lbq/inference_data
CODE_DIR=/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST


ranks="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

for rank in $ranks;
do
echo "Processing fr2_${rank} ... ..."
python ${CODE_DIR}/calculate_extended_components_new_fr2.py \
     --FIRSTcsv ${DATA_DIR}/first_14dec17.csv \
     --result ${RESULT_DIR}/FIRST_infer_part0-4_fr2_final.csv \
     --inpdir ${DATA_DIR}/FIRST_fits \
     --outdir ${RESULT_DIR} \
     --cls fr2 \
     --rank ${rank}

done




