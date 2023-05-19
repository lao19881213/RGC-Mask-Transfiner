#!/bin/bash

#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


#python3 setup.py build develop #--no-deps
#python3 setup.py develop #--no-deps


ID=159


#CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/train_net.py --num-gpus 3 --dist-url tcp://0.0.0.0:12346 \
#	--config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
#        --eval-only MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth \

CUDA_VISIBLE_DEVICES=0 python3 tools/train_net.py --num-gpus 1 --dist-url tcp://0.0.0.0:12346 \
        --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
        --eval-only MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth \

        #MODEL.RETINANET.SCORE_THRESH_TEST 0.5 \
        #MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.5 \
        #MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH 0.5
#./pretrained_model/output_3x_transfiner_r101_deform.pth


