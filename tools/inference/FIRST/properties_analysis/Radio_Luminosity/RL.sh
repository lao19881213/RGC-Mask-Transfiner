#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

labels="fr1"

for label in $labels;
do

python RL.py  --nedresult /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/NED/centre_peaks_5arcsec/info_${label}_flux.csv \
              --resultcsv /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv 

done 



