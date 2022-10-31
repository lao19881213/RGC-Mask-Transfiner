#!/bin/bash

source /home/blao/RGC-Mask-Transfiner/bashrc

srun -N 1 -p all-x86-cpu -w hw-x86-cpu01 python /home/blao/RGC-Mask-Transfiner/tools/sdc1/split_fits.py --fname /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/sdc1/data/SKAMid_B2_1000h_v3_train_image.fits --workdir /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/sdc1/data/split_B2_1000h
