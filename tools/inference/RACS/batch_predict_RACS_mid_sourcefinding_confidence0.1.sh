#!/bin/bash

cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

# export CUDA_LAUNCH_BLOCKING=1 # for debug

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part4_fits_fixed.txt"

DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"

for fn in `cat $IMG_FILES`;
do
SBID=${fn: 21:7}
if [ ! -f "/home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-mid_${SBID}_confidence0.1.csv" ];then

   CUDA_VISIBLE_DEVICES=0 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
     --input ${DATA_DIR}/${SBID}_split_fits_png \
     --pnglists ${DATA_DIR}/${SBID}_png.txt \
     --rmsdir ${DATA_DIR}/${SBID}_split_rms \
     --mirdir ${SBID}_split_mir \
     --output '/home/data0/lbq/RGC-Mask-Transfiner/RACS_results' \
     --confidence-threshold 0.1 \
     --catalogfn /home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-mid_${SBID}_confidence0.1 \
     --opts MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth
  
else
   echo "RACS-mid_${SBID}.csv already generated!"
fi
 
done                                                 
