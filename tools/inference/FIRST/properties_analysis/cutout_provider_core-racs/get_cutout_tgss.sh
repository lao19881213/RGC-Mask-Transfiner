#!/bin/bash


export PATH=/home/data0/lbq/software/anaconda3/bin:$PATH

python get_cutout_tgss.py --resultcsv /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv \
                      --outdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis 
