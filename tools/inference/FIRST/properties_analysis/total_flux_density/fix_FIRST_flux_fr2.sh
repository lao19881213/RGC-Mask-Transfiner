#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

clsns="fr2"

for clsn in ${clsns};
do
python3 fix_FIRST_flux.py \
        --recsv /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_${clsn}.csv \
        --outdir /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results 
done
