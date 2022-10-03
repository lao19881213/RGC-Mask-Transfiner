#!/bin/bash

source /home/blao/RGC-Mask-Transfiner/bashrc

python json2csv.py --inpdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask/fr2_add_2_bbox \
       --outdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/training_sets/v3_mask \
       --csvfn fr2_add_2.csv 
