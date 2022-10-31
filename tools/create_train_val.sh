#!/bin/bash

source /home/blao/RGC-Mask-Transfiner/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/data_prep_mrcnn.py --labelmejson '/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/train_final_aug/*.json' --outputjson /home/blao/RGC-Mask-Transfiner/datasets/coco/annotations/instances_train2022.json --version 'v3'

python /home/blao/RGC-Mask-Transfiner/tools/data_prep_mrcnn.py --labelmejson '/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/val_final/*.json' --outputjson /home/blao/RGC-Mask-Transfiner/datasets/coco/annotations/instances_val2022.json --version 'v3'