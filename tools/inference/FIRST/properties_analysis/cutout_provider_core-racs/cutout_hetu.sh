#!/bin/bash


export PATH=/home/data0/lbq/software/anaconda3/bin:$PATH
export PATH=/home/data0/lbq/software/Montage-6.0/bin:$PATH

python cutout_hetu.py --resultcsv /home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv \
                      --outdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis \
                      --surveys '0,2' 
