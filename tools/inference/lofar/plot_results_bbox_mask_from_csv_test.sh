#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=plot
##SBATCH --time=02:00:00
#SBATCH --error=err-%j.out
#SBATCH --partition=hw


source /home/blao/RGC-Mask-Transfiner/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/inference/lofar/plot_results_bbox_mask_from_csv.py \
       --resultsfile /home/blao/RGC-Mask-Transfiner/Mingo_no_sourcefinding.csv \
       --imagedir /p9550/LOFAR/LoTSS-DR1/Mingo_png_fr2_resize \
       --outdir /p9550/LOFAR/LoTSS-DR1/Mingo_hetu_test 


