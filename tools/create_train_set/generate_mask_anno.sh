#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python generate_mask_anno_cs.py --inpdir /home/data0/lbq/training_sets/cs_add_fits_png \
       --outdir /home/data0/lbq/training_sets/cs_add_anno \
       --csvfn /home/data0/lbq/training_sets/cs_add.csv 
