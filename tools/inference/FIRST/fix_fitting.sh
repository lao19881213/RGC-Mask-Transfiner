#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

cd /home/data0/lbq/RGC-Mask-Transfiner

python /home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/fix_fitting.py
