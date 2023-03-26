#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

labels="ht cj"

for label in $labels;
do

python ned_info_FIRST.py  /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv \
                         /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/NED \
                         $label

done 



