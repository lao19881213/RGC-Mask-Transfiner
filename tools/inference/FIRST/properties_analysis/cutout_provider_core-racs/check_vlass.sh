#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python check_vlass.py \
       --inpdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRI/VLASS \
       --outdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRI/VLASS_png

