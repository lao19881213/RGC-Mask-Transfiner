#!/bin/bash

#SBATCH --nodes=1
##SBATCH --time=16:00:00
#SBATCH --partition=hw
##SBATCH --export=ALL

#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

source /home/blao/RGC-Mask-Transfiner/bashrc

python3 check_data.py --input '/p9550/LOFAR/LoTSS-DR1/Mingo_png_fr2' \
  --pnglists '/p9550/LOFAR/LoTSS-DR1/mingo_png_fr2.txt' 
                                                  
