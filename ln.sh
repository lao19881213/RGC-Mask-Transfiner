#!/bin/bash

IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/image_cubes/fits_test.txt"


DATA_DIR="/home/data0/lbq/inference_data/ASKAP_RACS"


for fn in `cat $IMG_FILES`;
do
  rm -rf RACS-DR1_${fn}_split_mir
  ln -s /home/data0/lbq/inference_data/ASKAP_RACS/RACS-DR1_${fn}_split_mir RACS-DR1_${fn}_split_mir  
   
done
