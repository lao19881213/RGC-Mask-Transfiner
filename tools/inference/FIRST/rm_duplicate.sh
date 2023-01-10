#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
export PATH=/home/data0/lbq/software/topcat:$PATH

RESULT_DIR=/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results

echo "Processing FIRST catalog ... ..."
python /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/rm_duplicate.py --inpfn ${RESULT_DIR}/FIRST_infer_part0-4_th0.1_cs.csv --outdir ${RESULT_DIR}
