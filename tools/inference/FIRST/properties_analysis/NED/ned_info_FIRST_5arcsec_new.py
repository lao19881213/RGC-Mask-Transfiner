#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Automated download matched Ned data

import io
import os

import requests
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import sys
#from astroquery.ned import Ned
from astroquery.ipac.ned import Ned
import astropy.units as u
from astropy import coordinates
from astropy.io import fits
import astropy.wcs as wcs
#from mpi4py import MPI


csv_filename = sys.argv[1] 

#all_lines = np.loadtxt(csv_filename, comments='_', delimiter=',', dtype=str, usecols=(2))

#comm=MPI.COMM_WORLD
#num_process=comm.Get_size()
#rank=comm.Get_rank()

#Initialization
results_dir = sys.argv[2] #"/o9000/MWA/GLEAM-X/GLEAM-X-IDR1/HETU-GLEAM-X/catalogue/results-NED"

cln = sys.argv[3]

#if rank == 0:
if os.path.isdir(results_dir):
   print('Directory %s is already exists！' % results_dir)
else:
   os.makedirs(results_dir) 

#os.system("rm -rf %s/*.txt" % results_dir)


#comm.Barrier()

csv = pd.read_csv(csv_filename)

ra = csv['centre_ra'].values
dec = csv['centre_dec'].values
ra_peak = csv['ra'].values
dec_peak = csv['dec'].values
label = csv['label'].values
boxs = csv['box'].values
fns = csv['image_filename'].values
print(len(ra))

#pro_arr = np.array_split(np.arange(len(ra)),num_process)

tags = []
tags_non = []
#print(len(all_lines))
#print(all_lines)
#for i in range(len(ra)):
for i in range(len(ra)):#pro_arr[rank]:
    if label[i] == cln: 
       c1 = coordinates.SkyCoord(ra=ra[i], dec=dec[i], unit=(u.deg, u.deg), frame='fk5')
       c_new = c1.to_string('hmsdms', decimal=False, precision=1)
       c_new = c_new.replace('h','').replace('d','').replace('m','').replace('s','').replace(' ','')
       print(c_new)
       name = "J%s" % c_new
       print('[',i,'/',len(ra),']  ', name)
       fits_file = os.path.splitext(fns[i])[0] + ".fits" #fn.split('.')[0] + ".fits"
       hdu = fits.open(os.path.join("/home/data0/lbq/inference_data/FIRST_fits", fits_file))[0]
       w = wcs.WCS(hdu.header, naxis=2)
       x1 = float(boxs[i].split('-')[0])
       y1 = float(boxs[i].split('-')[1])
       x2 = float(boxs[i].split('-')[2])
       y2 = float(boxs[i].split('-')[3])   
       x1 = int(x1)
       y1 = int(y1)
       x2 = int(x2)
       y2 = int(y2)   
       box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
       box_data_ordered = sorted(box_data.flatten())
       #logger.info('%d, %d, %d, %d' % (hdu.data.shape[0]-y2, hdu.data.shape[0]-y1, x1,x2))
       peaks_ra = []
       peaks_dec = []
       peak_index = -2
       for m in range(10):
           peak_xy_offset = np.where(box_data==box_data_ordered[peak_index])
           if(np.size(peak_xy_offset)==0):
              peaks_ra.append(ra_peak[i])
              peaks_dec.append(dec_peak[i])
              peak_index = peak_index -1
              continue
           else:
              peak_x = x1 + peak_xy_offset[0][0]
           peak_y = hdu.data.shape[0]-y2 + peak_xy_offset[1][0]
           peak_ra, peak_dec = w.wcs_pix2world([[peak_x, peak_y]], 0)[0][0:2]
           if peak_ra < 0 :  #
              peak_ra = peak_ra + 360
           peaks_ra.append(peak_ra)
           peaks_dec.append(peak_dec)
           peak_index = peak_index -1
       ned_ras = [ra[i], ra_peak[i], peaks_ra[0], peaks_ra[1], \
                 peaks_ra[2], peaks_ra[3], peaks_ra[4], peaks_ra[5],\
                 peaks_ra[6], peaks_ra[7], peaks_ra[8], peaks_ra[9]]
       ned_decs = [dec[i], dec_peak[i], peaks_dec[0], peaks_dec[1], \
                 peaks_dec[2], peaks_dec[3], peaks_dec[4], peaks_dec[5],\
                 peaks_dec[6], peaks_dec[7], peaks_dec[8], peaks_dec[9]]

       cnt = 0
       except_cnt = 0
       for m in range(len(ned_ras)):
           try:
              co = coordinates.SkyCoord(ra=ned_ras[m], dec=ned_decs[m], unit=(u.deg, u.deg), frame='fk5')# fk5
              # FIRST resolution is 5 arcsec i.e. 0.00139 deg 
              result_table = Ned.query_region(co, radius=0.00139 * u.deg, equinox='J2000.0')# J2000
              #print(result_table)
              if(len(result_table['Object Name'])==0):
                  cnt = cnt + 1
                  continue
              else:
                  result_table.sort(['Separation', 'Redshift']) 
                  result_table_final = "{},{},{},{},{},{},{}".format(i,label[i],name,result_table['Object Name'][0],result_table['Type'][0],\
                                 result_table['Redshift'][0],result_table['Separation'][0])
                  for m in range(len(result_table)):
                      if result_table['Redshift'][m]>0:
                         result_table_final = "{},{},{},{},{},{},{}".format(i,label[i],name,result_table['Object Name'][m],result_table['Type'][m],\
                                 result_table['Redshift'][m],result_table['Separation'][m])
                         break
                  tags.append(result_table_final)
                  break
           except:
              except_cnt = except_cnt + 1
              continue        
           
           if(cnt==len(ned_ras) or except_cnt==len(ned_ras)): 
               print('Not found ', i)#name)
               tags_non.append("{},{},{},{},{}".format(i,label[i],name,ra[i],dec[i]))
        

#comm.Barrier()
resultsData = tags #comm.gather(tags, root=0)
resultsData_non = tags_non #comm.gather(tags_non, root=0)
#if rank==0 :
#resultsData = np.concatenate(resultsData,axis = 0)
#resultsData_non = np.concatenate(resultsData_non,axis = 0)
#print(resultsData)
with open(os.path.join(results_dir, 'info_%s.txt' % cln), 'w') as f:
     f.write(os.linesep.join(resultsData))
with open(os.path.join(results_dir, 'NOT_FOUND_NED_%s.txt' % cln), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))




