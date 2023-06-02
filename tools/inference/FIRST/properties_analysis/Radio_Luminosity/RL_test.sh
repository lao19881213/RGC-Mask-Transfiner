#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

labels="fr1"

for label in $labels;
do

python RL_test.py  --nedresult /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_${label}_optical_ned.csv 

done 



