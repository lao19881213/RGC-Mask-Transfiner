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
hdu= fits.open('FIRST_J001357-091951.fits')[0]
data = hdu.data[:,:]
img = hdu.data[:,:]
w1=wcs.WCS(hdu.header, naxis=2)
temp = fits.PrimaryHDU(data=data, header=w1.to_header())
pix_scale = abs(hdu.header['CDELT1'])

hdu_nvss= fits.open('NVSS_J001357.2-091951.5.fits')[0]
data_nvss = hdu_nvss.data[:,:]
img_nvss = hdu_nvss.data[:,:]
w1_nvss=wcs.WCS(hdu_nvss.header, naxis=2)
temp_nvss = fits.PrimaryHDU(data=data_nvss, header=w1_nvss.to_header())

hdu_vlass= fits.open('VLASS__3.1.ql.T08t01.J001357.17-091951.5.10.2048.v2.I.iter1.image.pbcor.tt0.subim_s2.0arcmin_.fits')[0]
data_vlass = hdu_vlass.data[0,0,:,:]
#print(hdu_vlass.data.shape)
img_vlass = hdu_vlass.data[0,0,:,:]
w1_vlass=wcs.WCS(hdu_vlass.header, naxis=2)
temp_vlass = fits.PrimaryHDU(data=data_vlass, header=w1_vlass.to_header())

#IR or Optical image
f=aplpy.FITSFigure('SDSS_J001357.2-091951.5.fits', slices=[0], north=True)
hdu_optical= fits.open('SDSS_J001357.2-091951.5.fits')[0]
data_optical = hdu_optical.data
#print(data_optical.shape)
f.show_grayscale(invert=True)
#f.show_grayscale() 
#f.show_colorscale(vmin=np.percentile(data_optical,5),vmax=np.percentile(data_optical,50), cmap='plasma')#,stretch='log')
#f.show_colorscale(vmax=21.0, cmap='gray_r')#cmap='plasma')#,stretch='log')
#f.show_colorscale(vmin=1e-1, cmap='gray_r',stretch='log')


rms =0.9* np.median(abs(img-np.median(img)))
#Plot the image in contours: 3 or 5
levs_positive = 5*rms*np.array([1,2,4,8,16,32,64])
#[1,np.sqrt(2),2,np.sqrt(2)*2,4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64,64*np.sqrt(2)])#,128,128*np.sqrt(2),256,256*np.sqrt(2)])
levs_negative = 5*rms*np.array([-1])

rms_nvss =0.9* np.median(abs(img_nvss-np.median(img_nvss)))
#Plot the image in contours: 3 or 5
levs_positive_nvss = 3*rms_nvss*np.array([1,2,4,8,16,32,64])#,128,128*np.sqrt(2),256,256*np.sqrt(2)])
levs_negative_nvss = 3*rms_nvss*np.array([-1])

rms_vlass =0.9* np.median(abs(img_vlass-np.median(img_vlass)))
#Plot the image in contours: 3 or 5
levs_positive_vlass = 5*rms_vlass*np.array([1,2,4,8,16,32,64])#,128,128*np.sqrt(2),256,256*np.sqrt(2)])
levs_negative_vlass = 3*rms_vlass*np.array([-1])

#FIRST magenta
#VLASS blue
#NVSS coral
f.show_contour(temp,dimensions=[0,1],colors='magenta',zorder=5,levels=levs_positive,slices=[0])#, alpha=0.3)
#f.show_contour(temp,dimensions=[0,1],colors='red',zorder=5,levels=levs_negative,linestyles='dashed',alpha=0.4)
f.show_contour(temp_vlass,dimensions=[0,1],colors='blue',zorder=5,levels=levs_positive_vlass,slices=[0])#, alpha=0.3)
f.show_contour(temp_nvss,dimensions=[0,1],colors='coral',zorder=5,levels=levs_positive_nvss,slices=[0])
#bounding box
boxs = '51.88499-34.50135-73.47312-78.67483'
x1 = float(boxs.split('-')[0])
y1 = float(boxs.split('-')[1])
x2 = float(boxs.split('-')[2])
y2 = float(boxs.split('-')[3])
width = x2 - x1
height = y2 - y1

radius = (max(width, height)/2.0 + 20) * pix_scale #deg
centre_ra = 3.48822
centre_dec = -9.33098195660072      
print(radius) 
f.recenter(centre_ra, centre_dec, radius=radius)


host_ra = 3.488499
host_dec = -9.330378

f.show_markers(host_ra, host_dec, edgecolor='c', facecolor='c',
                marker='x', s=100, zorder=6)#, alpha=-0.5)

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
f.save('%s.png' % 'J001357.2-091951.5')





