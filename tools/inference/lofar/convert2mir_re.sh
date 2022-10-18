#!/bin/bash

module use /home/software/modulefiles
module load MWA_Tools/cpu-mwa-sci

python convert2mir.py --fitsdir /p9550/LOFAR/LoTSS-DR1/re_fits --fitslist /p9550/LOFAR/LoTSS-DR1/mingo_fits_re.txt --mirdir /p9550/LOFAR/LoTSS-DR1/Mingo_mir 
