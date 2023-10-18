#!/bin/bash  

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

IMG_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"

#bmaj_fnl="9.0" #arcsec 0.0025 deg `echo "$bmin*3600.0" | bc -l`
#bmin_fnl="7.6" #arcsec 0.00211111111111111 deg
#bpa_fnl="71.5" #deg 
for id in {1..4};
do
IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part${id}_fits.txt"

for fn in `cat $IMG_FILES`;
do
    if [ ! -d "${IMG_DIR}/part${id}_conv" ];then
       mkdir -p ${IMG_DIR}/part${id}_conv
    else
       echo "${IMG_DIR}/part${id}_conv is already exists"
    fi
    echo "Processing ${fn} ... ..."
  
    if [ ! -f "${IMG_DIR}/part${id}_conv/${fn%%.fits}_fixed.fits" ]; then 
       cd ${IMG_DIR}/part${id}_conv  
       cp ${IMG_DIR}/part${id}/${fn} ${fn%%.fits}_fixed.fits
     fi
done
done

