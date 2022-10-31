#!/bin/bash  

#ssh -Y blao@202.127.3.157
 
source /home/blao/sdc1_resnet/bashrc

python /home/blao/sdc1_resnet/data/scripts/data_prep.py --rank=$1 --totalproc=$2
