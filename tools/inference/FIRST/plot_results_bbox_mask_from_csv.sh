#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=plot
##SBATCH --time=02:00:00
#SBATCH --error=err-%j.out
#SBATCH --partition=hw


source /home/blao/RGC-Mask-Transfiner/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/inference/FIRST/plot_results_bbox_mask_from_csv.py \
       --resultsfile /home/blao/RGC-Mask-Transfiner/FIRST_HT_infer.csv \
       --imagedir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT \
       --outdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT_hetu 


