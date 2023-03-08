#!/bin/bash



cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform_infer.yaml \
  --input '/home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_split_fits_png' \
  --pnglists '/home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_png.txt' \
  --rmsdir /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_split_rms \
  --mirdir RACS-DR1_0037-06A_split_mir \
  --output '/home/data0/lbq/RGC-Mask-Transfiner/RACS_results' \
  --confidence-threshold 0.5 \
  --catalogfn /home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-DR1_0037-06A \
  --opts MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth 
                                                  
