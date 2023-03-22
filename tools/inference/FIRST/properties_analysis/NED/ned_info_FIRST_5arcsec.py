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
label = csv['label'].values
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
       try:
           co = coordinates.SkyCoord(ra=ra[i], dec=dec[i], unit=(u.deg, u.deg), frame='fk5')# fk5
           # FIRST resolution is 5 arcsec i.e. 0.00139 deg 
           result_table = Ned.query_region(co, radius=0.00139 * u.deg, equinox='J2000.0')# J2000
           #print(result_table)
           result_table.sort(['Separation', 'Redshift'])
           result_table_final = "{},{},{},{},{},{},{}".format(i,label[i],name,result_table['Object Name'][0],result_table['Type'][0],\
                          result_table['Redshift'][0],result_table['Separation'][0])
           for m in range(len(result_table)):
               if result_table['Redshift'][m]>0:
                  result_table_final = "{},{},{},{},{},{},{}".format(i,label[i],name,result_table['Object Name'][m],result_table['Type'][m],\
                          result_table['Redshift'][m],result_table['Separation'][m])
                  break
           tags.append(result_table_final)
       except: #ValueError:
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




