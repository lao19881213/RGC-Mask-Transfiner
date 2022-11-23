#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python /home/data0/lbq/RGC-Mask-Transfiner/tools/data_prep_mrcnn.py --labelmejson '/home/data0/lbq/training_sets/train_final_aug/*.json' --outputjson /home/data0/lbq/RGC-Mask-Transfiner/datasets/coco/annotations/instances_train2022.json --version 'v3'

python /home/data0/lbq/RGC-Mask-Transfiner/tools/data_prep_mrcnn.py --labelmejson '/home/data0/lbq/training_sets/val_final/*.json' --outputjson /home/data0/lbq/RGC-Mask-Transfiner/datasets/coco/annotations/instances_val2022.json --version 'v3'
