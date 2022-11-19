#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python generate_mask_anno.py --inpdir  \
       --outdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/fr2_add_2_anno \
       --csvfn /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/fr2_add_2.csv 
