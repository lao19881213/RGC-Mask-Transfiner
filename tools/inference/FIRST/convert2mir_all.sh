#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH


python convert2mir.py --fitsdir /home/data0/lbq/inference_data/FIRST_fits --fitslist /home/data0/lbq/inference_data/FIRST.txt --mirdir /home/data0/lbq/inference_data/FIRST_mir
