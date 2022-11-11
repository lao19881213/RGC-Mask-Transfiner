#!/bin/bash  

#ssh -Y blao@202.127.3.157
 
source /home/blao/RGC-Mask-Transfiner/bashrc

python /home/blao/RGC-Mask-Transfiner/tools/sdc1/fits2png.py --rank=$1 --totalproc=$2
