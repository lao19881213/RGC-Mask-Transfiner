#!/bin/bash

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
export LD_LIBRARY_PATH=/home/gznu/anaconda3-2022/lib:$LD_LIBRARY_PATH

/usr/bin/Rscript /home/data0/lbq/RGC-Mask-Transfiner/tools/sky_model/ProFound/test.R
