#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python3 generate_bkg_files.py --inpdir /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/VLASS_final \
                              --outdir /home/data0/lbq/inference_data/HT/VLASS_bkg \
                              --fitslist /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/vlass_ht_fits.txt

