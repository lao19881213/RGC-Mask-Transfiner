#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=find
##SBATCH --time=02:00:00
#SBATCH --error=err-%j.out
#SBATCH --partition=hw

source /home/blao/rgz_resnet_fpn/bashrc

srun python /home/blao/rgz_resnet_fpn/tools/FIRST/rm_duplicate_fr12.py \
     --fr12csv /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/catalog/radio_IR/FIRST_all_final_objname_fr1.csv \
     --reptxt /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/catalog/radio_IR/FIRST_all_final_objname_fr1_rep.txt \
     --outdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/catalog/radio_IR \
     --outcsv FIRST_all_final_objname_fr1_rm_duplicate.csv



