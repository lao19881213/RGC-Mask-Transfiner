#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

CLASSES="FRII"

DATA_DIR="/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis"

OUT_DIR="/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis"

for cls in $CLASSES;
do
python fits2png_plt.py \
       --inpdir ${DATA_DIR}/${cls}/NVSS4flux \
       --outdir ${OUT_DIR}/${cls}/NVSS4flux_png

done

