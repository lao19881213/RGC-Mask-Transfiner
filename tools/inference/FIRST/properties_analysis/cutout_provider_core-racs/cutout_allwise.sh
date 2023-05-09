#!/bin/bash

export PATH=/home/data0/lbq/software/anaconda3-cutout/bin:$PATH
#export PATH=/home/data0/lbq/software/Montage-6.0/bin:$PATH

python3 cutout_allwise.py --resultcsv /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_cj_ALLWISE.csv \
                      --outdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis 
