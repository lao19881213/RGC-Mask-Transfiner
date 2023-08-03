#!/bin/bash

export PATH=/home/data0/lbq/software/topcat:$PATH

FIRST='/home/data0/lbq/inference_data/first_14dec17.csv'
MATCH='/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/first_14dec17_SDSS_DR16.csv'
stilts cdsskymatch cdstable=SDSS_DR16 \
                   in=$FIRST \
                   ra=RA dec=DEC radius=5 \
                   find=best out=$MATCH


FIRST='/home/data0/lbq/inference_data/first_14dec17.csv'
MATCH='/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/first_14dec17_PanSTARRS_DR1.csv'
stilts cdsskymatch cdstable=PanSTARRS_DR1 \
                   in=$FIRST \
                   ra=RA dec=DEC radius=5 \
                   find=best out=$MATCH
