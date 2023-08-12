#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

CLASSES="FRI FRII HT CJ"

DATA_DIR="/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis"

OUT_DIR="/home/data0/lbq/inference_data/FIRST_HeTu_png"

for cls in $CLASSES;
do
python fits2png_plt.py \
       --inpdir ${DATA_DIR}/${cls}/FIRST \
       --outdir ${OUT_DIR}/${cls}

done

