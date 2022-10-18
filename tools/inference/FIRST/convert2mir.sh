#!/bin/bash

module use /home/software/modulefiles
module load MWA_Tools/cpu-mwa-sci

python convert2mir.py --fitsdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT --fitslist /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT.txt --mirdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT_mir
