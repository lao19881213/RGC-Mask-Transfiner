#!/bin/bash


source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python plot_results_bbox_mask_from_csv.py \
       --resultsfile /home/data0/lbq/RGC-Mask-Transfiner/RACS_results/RACS-DR1_0037-06A.csv \
       --imagedir /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_split_fits_png \
       --outdir /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_0037-06A_pred 


