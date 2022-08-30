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

CUDA_VISIBLE_DEVICES=0 python3 demo/demo.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
  --input 'datasets/coco/val2022/*.png' \
  --output 'vis_hetu_r101_val/' \
  --opts MODEL.WEIGHTS ./output_101_3x_deform/model_0219999.pth #./pretrained_model/output_3x_transfiner_r50.pth
