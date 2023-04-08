#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
from astropy.utils.data import download_file
from astropy.io import fits  
from astropy.utils import data
import os
from astroquery.utils import TableList
from astropy.wcs import WCS
from astropy import wcs
from spectral_cube import SpectralCube as sc
import aplpy
import pandas as pd
import argparse

# In[16]:


#This step is mainly to process the data of the radio so that aplpy can read that data.

#parser = argparse.ArgumentParser()
#parser.add_argument('--resultf', help='hetu results file')
#parser.add_argument('--firstdir', help='input FIRST fits file directory')
#parser.add_argument('--wisedir', help='input WISE fits file directory')
#parser.add_argument('--outdir', help='output raido IR file directory')
#args = parser.parse_args()
#
#csv_re = pd.read_csv(args.resultf)
#ra = csv_re['centre_ra'].values
#dec = csv_re['centre_dec'].values
#
#objn = csv_re['object_name'].values
#
#first_dir = args.firstdir
#wise_dir = args.wisedir

#for m in range(len(objn)):
#    print("Processing %s ......" % objn[m])
#    fits_f = '%s.fits' % objn[m]
#    if(os.path.exists(os.path.join(wise_dir, fits_f))):
#radio image
#fig = plt.figure() 
hdu= fits.open('FIRST_J112443.8+384547.5.fits')[0]
data = hdu.data[:,:]
img = hdu.data[:,:]
w1=wcs.WCS(hdu.header, naxis=2)
temp = fits.PrimaryHDU(data=data, header=w1.to_header())


#IR or Optical image
f=aplpy.FITSFigure('SDSS_J112443.8+384547.5_r.fits', slices=[0], north=True)
hdu_optical= fits.open('SDSS_J112443.8+384547.5_r.fits')[0]
data_optical = hdu_optical.data
#print(data_optical.shape)
#f.show_grayscale() 
#f.show_colorscale(vmin=np.percentile(data_optical,5),vmax=np.percentile(data_optical,50), cmap='plasma')#,stretch='log')
f.show_colorscale(vmax=21.0, cmap='gray_r')#cmap='plasma')#,stretch='log')
rms =0.9* np.median(abs(img-np.median(img)))
#Plot the image in contours: 3 or 5
levs_positive = 3*rms*np.array([1,np.sqrt(2),2,np.sqrt(2)*2,4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64,64*np.sqrt(2),128,128*np.sqrt(2),256,256*np.sqrt(2)])
levs_negative = 3*rms*np.array([-1])
f.show_contour(temp,dimensions=[0,1],colors='black',zorder=5,levels=levs_positive,slices=[0])
f.show_contour(temp,dimensions=[0,1],colors='red',zorder=5,levels=levs_negative,linestyles='dashed',alpha=0.4)
#Here need to add the position of sources(RA,DEC)(radio) in catalogue, pixel scale 1.8 asec
f.recenter(171.18248,38.76319158543373, radius=0.009)

f.axis_labels.set_xtext('Right Ascension (J2000)')
f.axis_labels.set_ytext('Declination (J2000)')
f.axis_labels.set_font(size=14, weight='medium', stretch='normal', family='sans-serif', style='normal', variant='normal')
#plt.xlabel('Right Ascension (J2000)',fontsize=14)#'xx-large')
#plt.ylabel('Declination (J2000)',fontsize=14)#'xx-large')

plt.tick_params(axis='both',direction='in',color='black', labelsize=14)
f.add_colorbar()
f.colorbar.set_width(0.2)
f.colorbar.set_location('right')
#f.colorbar.set_axis_label_text('Jy/beam')
f.colorbar.set_pad(0.1)
f.colorbar.set_axis_label_font(size=14)#'x-large')

f.tick_labels.set_font(size=14, weight='medium', stretch='normal', family='sans-serif', style='normal', variant='normal')
#plt.tight_layout()
#the name of file need to change
#f.save()
f.save('%s.png' % 'J112443.8+384547.5')


# In[ ]:





# In[ ]:




