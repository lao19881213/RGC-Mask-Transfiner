#!/bin/bash

#SBATCH --nodes=1
##SBATCH --time=16:00:00
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200g
#SBATCH --partition=sugon-gpu
##SBATCH --export=ALL

#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

cd /home/blao/RGC-Mask-Transfiner

source /home/blao/RGC-Mask-Transfiner/bashrc

export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=0 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
  --input '/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT_png' \
  --pnglists '/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT_png.txt' \
  --output '/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT_pred' \
  --confidence-threshold 0.7 \
  --nosourcefinding \
  --catalogfn FIRST_HT \
  --opts MODEL.WEIGHTS ./output_101_3x_deform_v1/model_0219999.pth 
  #./pretrained_model/output_3x_transfiner_r50.pth
                                                  
