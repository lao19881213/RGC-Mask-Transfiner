#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu

declare -A IMGS

#SNRS="5 10 20 30 50 100 200"
IMG_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"
#fns="SB9287 SB9325 SB9351 SB9410 SB9434 SB9437 SB9442 SB9501 SB10083 SB10635"
#IMGS=([SB9287]='0.0000268' [SB9325]='0.0000255' [SB9351]='0.0000295' [SB9410]='0.0000258' [SB9434]='0.0000216' [SB9437]='0.0000258' [SB9442]='0.0000239' [SB9501]='0.0000246' [SB10083]='0.0000233' [SB10635]='0.0000248')

IMGS="0037-06A"

for fn in ${IMGS};
do
    #for snr in $SNRS;
    #do
    if [ ! -d "${IMG_DIR}/RACS-DR1_${fn}_split_png" ];then
       mkdir -p ${IMG_DIR}/RACS-DR1_${fn}_split_png
    else
       echo "${IMG_DIR}/RACS-DR1_${fn}_split_png is already exists"
    fi
    echo "Processing ${fn} ... ..."

    python fits2png_RACS_plt.py --inpdir ${IMG_DIR}/RACS-DR1_${fn}_split_fits --outdir ${IMG_DIR}/RACS-DR1_${fn}_split_png 
    #done
done

