#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/fits2png_FIRST_plt.py \
       --inpdir /home/data0/lbq/inference_data/nomatch_cs_hetu \
       --outdir /home/data0/lbq/inference_data/nomatch_cs_hetu_png

