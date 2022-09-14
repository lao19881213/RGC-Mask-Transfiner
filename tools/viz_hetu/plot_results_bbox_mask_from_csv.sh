#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=plot
##SBATCH --time=02:00:00
#SBATCH --error=err-%j.out
#SBATCH --partition=hw


source /home/blao/RGC-Mask-Transfiner/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/viz_hetu/plot_results_bbox_mask_from_csv.py \
       --resultsfile /home/blao/RGC-Mask-Transfiner/lofar_test.csv \
       --imagedir /home/blao/RGC-Mask-Transfiner/datasets/coco/LOFAR \
       --outdir /home/blao/RGC-Mask-Transfiner/vis_hetu_lofar 


