import os
import numpy as np
import xml.etree.ElementTree as ET
import pdb
from astropy.io import fits
import astropy.wcs as wcs
from PIL import Image
import matplotlib as mpl
import argparse
from mpi4py import MPI
import pandas as pd
import linecache


result_file = 'results.csv' 


input_dir = './'
#file_nms = os.listdir(input_dir)
comm=MPI.COMM_WORLD
num_process=comm.Get_size()
rank=comm.Get_rank()

FIRST_csv = '/home/data0/lbq/inference_data/first_14dec17.csv'

csv_FIRST = pd.read_csv(FIRST_csv)

ra_FIRST = csv_FIRST['RA'].values

dec_FIRST = csv_FIRST['DEC'].values

#uuid = csv_FIRST['index'].values

csv=pd.read_csv(result_file)
image_file = csv['imagefilename'].values
box = csv['box'].values
predicted_class = csv['label'].values
#ra = csv['centre_ra'].values
#dec = csv['centre_dec'].values


#print(csv.loc[0]['box'], len(csv))

tags = []
tags_first = []

pro_arr = np.array_split(np.arange(len(image_file)),num_process)
for n in pro_arr[rank]:
   fits_file = os.path.splitext(image_file[n])[0] + ".fits"
   hdu = fits.open(os.path.join(input_dir, fits_file))
   image = hdu[0].data
   w = wcs.WCS(hdu[0].header, naxis=2)
   width = hdu[0].data.shape[1]
   height = hdu[0].data.shape[0]

   bottom_left = [0, 0]
   top_left = [0, height - 1]
   top_right = [width - 1, height - 1]
   bottom_right = [width - 1, 0]

   ret = np.zeros([4, 2])
   ret[0, :] = w.wcs_pix2world([bottom_left], 0)[0][0:2] 
   ret[1, :] = w.wcs_pix2world([top_left], 0)[0][0:2] 
   ret[2, :] = w.wcs_pix2world([top_right], 0)[0][0:2] 
   ret[3, :] = w.wcs_pix2world([bottom_right], 0)[0][0:2] 
   RA_min, DEC_min, RA_max, DEC_max = np.min(ret[:, 0]),   np.min(ret[:, 1]),  np.max(ret[:, 0]),  np.max(ret[:, 1])

   x1 = float(box[n].split('-')[0])
   y1 = float(box[n].split('-')[1])
   x2 = float(box[n].split('-')[2])
   y2 = float(box[n].split('-')[3])

   x1 = int(x1)
   y1 = int(y1)
   x2 = int(x2)
   y2 = int(y2)
   print(x1,x2,y1,y2)
   
   y1_new = hdu[0].data.shape[0]-y2
   y2_new = hdu[0].data.shape[0]-y1
   centre_y = int(y1_new + (y2_new - y1_new)/2)
   centre_x = int(x1 + (x2 - x1)/2)
   centre_ra, centre_dec = w.wcs_pix2world([[centre_x, centre_y]], 0)[0][0:2]
   #centre_ra, centre_dec = w.all_pix2world(y1_new+1, x1+1, 1)
   ra_re = centre_ra
   dec_re = centre_dec
   
   label = '{}'.format(predicted_class[n])

   top, left, bottom, right = y1_new, x1, y2_new, x2

   bottom_left_obj = [x1, y1_new]
   top_left_obj = [x1, y2_new]
   top_right_obj = [x2, y2_new]
   bottom_right_obj = [x2, y1_new]

   ret_obj = np.zeros([4, 2])
   ret_obj[0, :] = w.wcs_pix2world([bottom_left_obj], 0)[0][0:2]
   ret_obj[1, :] = w.wcs_pix2world([top_left_obj], 0)[0][0:2]
   ret_obj[2, :] = w.wcs_pix2world([top_right_obj], 0)[0][0:2]
   ret_obj[3, :] = w.wcs_pix2world([bottom_right_obj], 0)[0][0:2]
   RA_min_obj, DEC_min_obj, RA_max_obj, DEC_max_obj = np.min(ret_obj[:, 0]),   np.min(ret_obj[:, 1]),  np.max(ret_obj[:, 0]),  np.max(ret_obj[:, 1])
   print(label, (left, top), (right, bottom))
   print(ra_re)
   comp_cnt = 0
    
   for m in range(len(ra_FIRST)):
       ra_first = float(ra_FIRST[m])
       dec_first = float(dec_FIRST[m])
       img_x, img_y = w.wcs_world2pix([[ra_first,dec_first]],0).transpose()
       if ra_first <= RA_max and ra_first >= RA_min and dec_first <= DEC_max and dec_first >= DEC_min:
          if img_x <= x2 and img_x >= x1 and img_y <= y2_new and img_y >= y1_new: 
             comp_cnt = comp_cnt + 1
             tags_first.append('{},{},{}'.format(image_file[n], ra_first, dec_first))


outdir = './'

comm.Barrier()
firstdata = comm.gather(tags_first, root=0)
if rank==0 :
   firstdata = np.concatenate(firstdata,axis = 0)
   #print(resultsData)
   with open(os.path.join(outdir, 'FIRST_extended_components.txt'), 'w') as f:
        f.write(os.linesep.join(firstdata))
