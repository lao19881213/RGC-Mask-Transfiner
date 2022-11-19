#!/bin/bash


#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

#export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159

CUDA_VISIBLE_DEVICES=1,2 python3 tools/plain_train_net.py --num-gpus 2 --resume --dist-url tcp://0.0.0.0:12349\
	--config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml 

