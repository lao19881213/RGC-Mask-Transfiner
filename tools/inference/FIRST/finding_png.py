import pandas as pd
import os
from mpi4py import MPI
import linecache
import numpy as np

csv = pd.read_csv('/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/hetu_results/final/FIRST_all_infer.csv')

img_fn = csv['imagefilename']

label = csv['label']

comm=MPI.COMM_WORLD
num_process=comm.Get_size()
rank=comm.Get_rank()

labels = ['cs','fr2','fr1','cj', 'ht']
#if rank==0:
#    for m in range(len(img_fn)):
#        for labeln in labels:
#            if label[m] == labeln:
#               os.system('cp /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/FIRST_fits/%s /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/hetu_results/%s' %(img_fn[m],labeln))


fits_list = "/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/FIRST_final.txt"
with open(fits_list, 'r') as file:
     linecount=len(file.readlines())

#file_nms = os.listdir('/o9000/MWA/GLEAM/hetu_images/deep_learn/inference_sets/FIRST_fits')

img_fn = img_fn.values

pro_arr = np.array_split(np.arange(linecount),num_process)
for n in pro_arr[rank]:
    line = linecache.getline(fits_list, n+1)
    fits_file = line.split('\n')[0]
    fn = os.path.splitext(fits_file)[0] + ".png"
    if fn in img_fn:
       print('%s already detected!' % fn)
    else:
       os.system('cp /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/FIRST_fits/%s /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/hetu_results/final/no_detect_png' % fn)
       os.system('cp /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/FIRST_fits/%s /p9550/MWA/GLEAM/blao/hetu_images/deep_learn/inference_sets/hetu_results/final/no_detect_fits' % fits_file)
