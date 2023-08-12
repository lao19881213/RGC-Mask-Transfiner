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


csv_filename = sys.argv[1] 

#Initialization
results_dir = sys.argv[2] #

cln = sys.argv[3]

if os.path.isdir(results_dir):
   print('Directory %s is already exists！' % results_dir)
else:
   os.makedirs(results_dir) 


csv = pd.read_csv(csv_filename)

ra = csv['centre_ra'].values
dec = csv['centre_dec'].values
ra_peak = csv['ra'].values
dec_peak = csv['dec'].values
label = csv['label'].values
boxs = csv['box'].values
fns = csv['image_filename'].values
print(len(ra))

tags = []
tags_non = []
for i in range(len(ra)):
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
          result_table.sort(['Separation']) 
          result_table_final = "{},{},{},{},{},{},{}".format(i,label[i],name,result_table['Object Name'][0],result_table['Type'][0],\
                             result_table['Redshift'][0],result_table['Separation'][0])
          tags.append(result_table_final)
       except:
          print('Not found ', i)#name)
          tags_non.append("{},{},{},{},{}".format(i,label[i],name,ra[i],dec[i]))
          continue        
           

resultsData = tags 
resultsData_non = tags_non
 
with open(os.path.join(results_dir, 'info_%s.txt' % cln), 'w') as f:
     f.write(os.linesep.join(resultsData))
with open(os.path.join(results_dir, 'NOT_FOUND_NED_%s.txt' % cln), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))




