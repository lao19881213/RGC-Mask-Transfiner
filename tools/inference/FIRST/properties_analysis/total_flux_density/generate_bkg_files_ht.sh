#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python3 generate_bkg_files.py --inpdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/NVSS \
                              --outdir /home/data0/lbq/inference_data/HT/NVSS_bkg \
                              --fitslist /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/nvss_fr2.txt

