#!/bin/bash

#SBATCH --nodes=1
##SBATCH --time=16:00:00
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=200g
#SBATCH --partition=inspur-gpu-ib
##SBATCH --export=ALL

#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

cd /home/blao/RGC-Mask-Transfiner

source /home/blao/RGC-Mask-Transfiner/bashrc
#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/plain_train_net.py --num-gpus 8 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml 

