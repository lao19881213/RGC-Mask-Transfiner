#!/bin/bash

source /home/blao/rgz_resnet_fpn/bashrc
module load topcat/v4.8

RESULT_DIR=/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/hetu_results/final

echo "Processing FIRST catalog ... ..."
python /home/blao/rgz_resnet_fpn/tools/FIRST/rm_duplicate.py --inpfn ${RESULT_DIR}/FIRST_all_infer.csv --outdir ${RESULT_DIR}
