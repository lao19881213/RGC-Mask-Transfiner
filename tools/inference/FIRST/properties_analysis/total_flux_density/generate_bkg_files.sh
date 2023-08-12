#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python3 generate_bkg_files.py --inpdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/NVSS \
                              --outdir /home/data0/lbq/inference_data/NVSS_bkg \
                              --fitslist /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII/nvss_fr2.txt

