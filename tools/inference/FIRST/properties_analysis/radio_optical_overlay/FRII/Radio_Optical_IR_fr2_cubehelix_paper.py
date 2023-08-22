#!/usr/bin/env python
# coding: utf-8

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
import cubehelix
from astropy.coordinates import SkyCoord
from astropy import units as u

paper_list = '/home/data0/lbq/inference_data/radio_optical/fr2_paper.txt'

out_dir = '/home/data0/lbq/inference_data/radio_optical/FRII_paper'
data_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII'
FIRST_dir = '%s/FIRST' % data_dir
NVSS_dir = '%s/NVSS' % data_dir
VLASS_dir = '%s/VLASS_final' % data_dir
SDSS_dir = '%s/SDSS' % data_dir

resultf = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_SDSS_DR16.csv'

csv_re = pd.read_csv(resultf)
ras = csv_re['centre_ra'].values
decs = csv_re['centre_dec'].values
bboxs = csv_re['box'].values
objns = csv_re['source_name'].values
sdss_ras = csv_re['RA_ICRS'].values
sdss_decs = csv_re['DE_ICRS'].values 
sdss_names = csv_re['SDSS16'].values
vlass_fitsns = os.listdir(VLASS_dir)

#
with open(paper_list, 'r') as f:
     for line in f:
         sn = line.rstrip('\n')
         for m in range(len(objns)):
             if sn == objns[m]:
                print("Processing %s ......" % objns[m])
                FIRST_fits_f = '%s/%s.fits' % (FIRST_dir, objns[m])
                #if(os.path.exists(os.path.join(wise_dir, fits_f))):
                hdu= fits.open(FIRST_fits_f)[0]
                data = hdu.data[:,:]
                img = hdu.data[:,:]
                w1=wcs.WCS(hdu.header, naxis=2)
                temp = fits.PrimaryHDU(data=data, header=w1.to_header())
                pix_scale = abs(hdu.header['CDELT1'])
                
                #IR or Optical image
                sdss_fits_f = '%s/SDSS_%s_r.fits' % (SDSS_dir, objns[m])
                f=aplpy.FITSFigure(sdss_fits_f, slices=[0], north=True)
                hdu_optical= fits.open(sdss_fits_f)[0]
                data_optical = hdu_optical.data
                f.show_colorscale(vmin=0.0, smooth=3, cmap=cubehelix.cmap(reverse=True))
                
                rms =0.9* np.median(abs(img-np.median(img)))
                #Plot the image in contours: 3 or 5
                levs_positive = 3*rms*np.array([1,np.sqrt(2),2,np.sqrt(2)*2,4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64,64*np.sqrt(2),128,128*np.sqrt(2),256,256*np.sqrt(2)])
                levs_negative = 3*rms*np.array([-1])
                
                f.show_contour(temp,dimensions=[0,1],colors='red',zorder=5,levels=levs_positive,slices=[0])#, alpha=0.3)
                #bounding box
                boxs = bboxs[m]
                x1 = float(boxs.split('-')[0])
                y1 = float(boxs.split('-')[1])
                x2 = float(boxs.split('-')[2])
                y2 = float(boxs.split('-')[3])
                width = x2 - x1
                height = y2 - y1
                
                radius = (max(width, height)/2.0 + 4) * pix_scale #deg
                centre_ra = ras[m]
                centre_dec = decs[m]
                print(radius) 
                f.recenter(centre_ra, centre_dec, radius=radius)
                
                
                host_ra = sdss_ras[m]
                host_dec = sdss_decs[m]
                host_icrs = SkyCoord(ra=host_ra*u.degree, dec=host_dec*u.degree, frame='icrs') 
                host_fk5 = host_icrs.transform_to('fk5')
                f.show_markers(host_fk5.ra.value, host_fk5.dec.value, edgecolor='b', facecolor='b',
                                marker='x', s=100, zorder=6)#, alpha=-0.5)
                
                source_name = sdss_names[m].split(' ')[1][0:5] + sdss_names[m].split(' ')[1][10:15]
                f.set_title(source_name, size=24, fontname = 'serif') 
                f.axis_labels.set_xtext('Right Ascension (J2000)')
                f.axis_labels.set_ytext('Declination (J2000)')
                f.axis_labels.set_font(size=28, weight='medium', stretch='normal', family='sans-serif', style='normal', variant='normal')
                #plt.xlabel('Right Ascension (J2000)',fontsize=14)#'xx-large')
                #plt.ylabel('Declination (J2000)',fontsize=14)#'xx-large')
                
                plt.tick_params(which='major', length=8)
                plt.tick_params(which='minor', length=4) 
                plt.tick_params(axis='both',direction='in',color='black')
                #f.add_colorbar()
                #f.colorbar.set_width(0.2)
                #f.colorbar.set_location('right')
                ##f.colorbar.set_axis_label_text('Jy/beam')
                #f.colorbar.set_pad(0.1)
                #f.colorbar.set_axis_label_font(size=14)#'x-large')
                
                f.tick_labels.set_font(size=24, weight='medium', stretch='normal', family='sans-serif', style='normal', variant='normal')
                #plt.tight_layout()
                #the name of file need to change
                #f.save()
                f.save('%s/%s_optical_radio_cubehelix.png' % (out_dir, objns[m]))
                f.save('%s/%s_optical_radio_cubehelix.pdf' % (out_dir, objns[m]))
                #else:
                #   print('VLASS image not found!')





