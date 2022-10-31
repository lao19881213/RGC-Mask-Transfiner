#!/bin/bash
source /home/blao/sdc1_resnet/bashrc 
srun -N 1 -p purley-cpu -w purley-x86-cpu01 python create_xml_anno.py
