#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=plot
##SBATCH --time=02:00:00
#SBATCH --error=err-%j.out
#SBATCH --partition=hw


source /home/blao/rgz_resnet_fpn/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/viz_hetu/plot_results_bbox_mask.py 


