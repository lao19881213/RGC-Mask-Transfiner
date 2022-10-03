#!/bin/bash

source /home/blao/RGC-Mask-Transfiner/bashrc

python generate_mask_anno.py --inpdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/fr2_add_2_fits_png \
       --outdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/fr2_add_2_anno \
       --csvfn /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/fr2_add_2.csv 
