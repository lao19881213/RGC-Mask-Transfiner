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
#python3 setup.py develop #--no-deps

export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug

ID=159


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tools/train_net.py --num-gpus 8 --dist-url tcp://0.0.0.0:12346 \
	--config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
        --eval-only MODEL.WEIGHTS ./output_101_3x_deform/model_0219999.pth \
        MODEL.RETINANET.SCORE_THRESH_TEST 0.7 \
        MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.7 \
        MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH 0.7
#./pretrained_model/output_3x_transfiner_r101_deform.pth


