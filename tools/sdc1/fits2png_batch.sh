#!/bin/bash  

#ssh -Y blao@202.127.3.157
 
#for i in {01..3};
#do
#   sbatch -N 1 -p all-x86-cpu -w hw-x86-cpu$i ./data_prep.sh $i 3 
#done

sbatch -N 1 -p all-x86-cpu -w hw-x86-cpu01 ./fits2png.sh 1 1
