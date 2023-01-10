#!/bin/bash


cd /home/data0/lbq/RGC-Mask-Transfiner

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu


CUDA_VISIBLE_DEVICES=0 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform.yaml \
  --input '/home/data0/lbq/RGC-Mask-Transfiner/tools/association_components' \
  --pnglists '/home/data0/lbq/RGC-Mask-Transfiner/tools/association_components/test.txt' \
  --output '/home/data0/lbq/RGC-Mask-Transfiner/tools/association_components' \
  --confidence-threshold 0.7 \
  --nosourcefinding \
  --catalogfn /home/data0/lbq/RGC-Mask-Transfiner/tools/association_components/results \
  --opts MODEL.WEIGHTS ./output_101_3x_deform/model_0229999.pth 
  #./pretrained_model/output_3x_transfiner_r50.pth
                                                  
