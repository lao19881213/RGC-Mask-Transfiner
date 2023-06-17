#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python check_vlass.py \
       --inpdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/VLASS_add \
       --outdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/VLASS_add_png

