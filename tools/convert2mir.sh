#!/bin/bash

module use /home/software/modulefiles
module load MWA_Tools/cpu-mwa-sci

python convert2mir.py --fitsdir /home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR --fitslist /home/blao/RGC-Mask-Transfiner/datasets/coco/lofar.txt --mirdir /home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR_mir 
