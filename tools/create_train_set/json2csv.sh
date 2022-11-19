#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

python json2csv.py --inpdir /home/data0/lbq/training_sets/cs_add_bbox \
       --outdir /home/data0/lbq/training_sets \
       --csvfn cs_add.csv 
