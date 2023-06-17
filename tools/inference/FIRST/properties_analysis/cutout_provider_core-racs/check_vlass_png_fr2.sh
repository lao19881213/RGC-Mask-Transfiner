#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python check_vlass_png.py \
       --catalogfile /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2.csv \
       --pngdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/VLASS_fr2_final \
       --cln fr2

