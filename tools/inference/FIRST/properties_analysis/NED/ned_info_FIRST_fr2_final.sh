#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

labels="fr2"

for label in $labels;
do

python ned_info_FIRST_fr2_final.py  /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_${label}_flux_fixed_vlass.csv \
                         /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/NED/${label} \
                         $label

done 



