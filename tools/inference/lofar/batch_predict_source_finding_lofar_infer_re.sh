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
module use /home/software/modulefiles
module load miriad/cpu-2007

export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug

CUDA_VISIBLE_DEVICES=1 python3 demo/batch_predict.py --config-file configs/transfiner/mask_rcnn_R_101_FPN_3x_deform_infer.yaml \
  --input '/p9550/LOFAR/LoTSS-DR1/re_fits' \
  --pnglists '/p9550/LOFAR/LoTSS-DR1/mingo_png_re.txt' \
  --mirdir 'Mingo_mir' \
  --rmsdir '/p9550/LOFAR/LoTSS-DR1/Mingo_rms' \
  --output '/p9550/LOFAR/LoTSS-DR1/Mingo_hetu/' \
  --confidence-threshold 0.1 \
  --catalogfn Mingo_re \
  --opts MODEL.WEIGHTS ./output_101_3x_deform_infer/model_0249999.pth 
  #./pretrained_model/output_3x_transfiner_r50.pth
                                                  