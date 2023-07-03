#!/bin/bash

#SBATCH --partition=hw
#SBATCH --job-name=match
#SBATCH --nodes=1
#SBATCH --mem=60gb

module use /home/app/modulefiles
module load topcat/v4.8

proxy=http://192.168.6.123:3128

FIRST='/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/catalog/non_match_hetu_final_cs.csv'
MATCH='/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/catalog/non_match_hetu_final_cs_allwise_match.csv'
stilts cdsskymatch cdstable=ALLWISE \
                   in=$FIRST \
                   ra=ra dec=dec radius=5 \
                   find=best out=$MATCH
