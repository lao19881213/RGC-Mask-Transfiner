#!/bin/bash

cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

# export CUDA_LAUNCH_BLOCKING=1 # for debug

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"

DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"

for fn in `cat $IMG_FILES`;
do
if [ ! -f "/home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-DR1_${fn}.csv" ];then

   CUDA_VISIBLE_DEVICES=0 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
     --input ${DATA_DIR}/RACS-DR1_${fn}_split_fits_png \
     --pnglists ${DATA_DIR}/RACS-DR1_${fn}_png.txt \
     --rmsdir ${DATA_DIR}/RACS-DR1_${fn}_split_rms \
     --mirdir RACS-DR1_${fn}_split_mir \
     --output '/home/data0/lbq/RGC-Mask-Transfiner/RACS_results' \
     --confidence-threshold 0.5 \
     --catalogfn /home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-DR1_${fn} \
     --opts MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth
  
else
   echo "RACS-DR1_${fn}.csv already generated!"
fi
 
done                                                 
