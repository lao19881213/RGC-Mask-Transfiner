#!/bin/bash


#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

# export CUDA_LAUNCH_BLOCKING=1 # for debug

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CUDA_VISIBLE_DEVICES=1 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform_infer.yaml \
  --input '/home/data0/lbq/inference_data/FIRST_fits' \
  --pnglists '/home/data0/lbq/inference_data/FIRST_final_png_part4.txt' \
  --rmsdir /home/data0/lbq/inference_data/FIRST_rms \
  --mirdir FIRST_mir \
  --output '/home/data0/lbq/inference_data/FIRST_pred' \
  --confidence-threshold 0.1 \
  --catalogfn FIRST_infer_part4_th0.1 \
  --opts MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth                                       
