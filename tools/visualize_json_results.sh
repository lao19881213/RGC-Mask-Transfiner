#!/bin/bash

#SBATCH --nodes=1
##SBATCH --time=16:00:00
##SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=200g
#SBATCH --partition=sugon-gpu
##SBATCH --export=ALL

#python3 setup.py build develop #--no-deps
# python3 setup.py develop #--no-deps

cd /home/blao/RGC-Mask-Transfiner

source /home/blao/RGC-Mask-Transfiner/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/visualize_json_results.py \
--input /home/blao/RGC-Mask-Transfiner/output_101_3x_deform/inference/coco_2022_val/coco_instances_results.json \
--output /home/blao/RGC-Mask-Transfiner/vis_hetu \
--dataset coco_2022_val \
--conf-threshold 0.7
