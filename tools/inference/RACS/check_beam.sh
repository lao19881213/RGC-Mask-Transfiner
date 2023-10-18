#!/bin/bash  

source /home/data0/lbq/RGC-Mask-Transfiner/bashrc_gznu
#declare -A IMGS

for id in {1..4};
do
    echo "processing part${id} ... ..."
    python  check_beam.py --partid ${id}
done

