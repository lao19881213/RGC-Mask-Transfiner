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

paper_list = '/home/data0/lbq/inference_data/radio_optical/GRGs.txt'

out_dir = '/home/data0/lbq/inference_data/radio_optical/GRGs_paper'
data_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/FRII'
FIRST_dir = '%s/FIRST' % data_dir
NVSS_dir = '%s/NVSS' % data_dir
VLASS_dir = '%s/VLASS_final' % data_dir
SDSS_dir = '%s/SDSS' % data_dir

resultf = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_sdss_ned_flux_fixed_vlass_optical_spectra.csv' #FIRST_HeTu_paper_fr2_SDSS_DR16_flux_fixed_vlass.csv'

csv_re = pd.read_csv(resultf)
ras = csv_re['centre_ra'].values
decs = csv_re['centre_dec'].values
bboxs = csv_re['box'].values
objns = csv_re['source_name'].values
sdss_ras = csv_re['sdss_ra'].values #['RA_ICRS'].values
sdss_decs = csv_re['sdss_dec'].values #['DE_ICRS'].values 
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
                img = np.nan_to_num(data)
                w1=wcs.WCS(hdu.header, naxis=2)
                data = np.nan_to_num(data)
                temp = fits.PrimaryHDU(data=data, header=w1.to_header())
                pix_scale = abs(hdu.header['CDELT1'])

                NVSS_fits_f = '%s/NVSS_%s.fits' % (NVSS_dir, objns[m])
                hdu_nvss= fits.open(NVSS_fits_f)[0]
                data_nvss = hdu_nvss.data[:,:]
                img_nvss = hdu_nvss.data[:,:]
                w1_nvss=wcs.WCS(hdu_nvss.header, naxis=2)
                temp_nvss = fits.PrimaryHDU(data=data_nvss, header=w1_nvss.to_header())

                items = objns[m]
                #print(items)
                if(len(items.split('-'))>=2):
                  RAlist=items.split('-')[0].split('J')[1]
                  Declist=items.split('-')[1]
                  ra_first = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
                  dec_first = 0 - (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
                elif(len(items.split('+'))>=2):
                  RAlist=items.split('+')[0].split('J')[1]
                  Declist=items.split('+')[1]
                  ra_first = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
                  dec_first = (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
                print(ra_first, dec_first)
                #sky_first = SkyCoord.from_name(items)
                #ra_first = sky_first.ra.value
                #dec_first = sky_first.dec.value
                #print(ra_first, dec_first)
                for n in range(len(vlass_fitsns)):
                    fn = 'J' + vlass_fitsns[n].split('J')[1]
                    pngn = fn[0:19]
                    # J072959.65+282815.6
                    if(len(pngn.split('-'))>=2):
                      RAlist=pngn.split('-')[0].split('J')[1]
                      Declist=pngn.split('-')[1]
                      ra_vlass = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
                      dec_vlass = 0 - (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
                    elif(len(pngn.split('+'))>=2):
                      RAlist=pngn.split('+')[0].split('J')[1]
                      Declist=pngn.split('+')[1]
                      ra_vlass = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
                      dec_vlass = (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0) 
                    #sky_vlass = SkyCoord.from_name(pngn)
                    #ra_vlass = sky_vlass.ra.value
                    #dec_vlass = sky_vlass.dec.value
                     
                    if((abs((ra_first-ra_vlass)*3600.0) <=1.5) and (abs((dec_first-dec_vlass)*3600.0) <=1.5)):
                       #print(pngn)
                       #print((ra_first-ra_vlass)*3600.0, dec_first - dec_vlass)
                       #print('Cross matched FIRST and VLASS: %s, %s' % (items, pngn))
                       vlass_fits_f = vlass_fitsns[n]
                       break
                    else:
                       if n == len(vlass_fitsns) - 1:
                           vlass_fits_f = ''
                if vlass_fits_f != '':
                   print(vlass_fits_f) 
                   hdu_vlass= fits.open('%s/%s' % (VLASS_dir, vlass_fits_f))[0]
                   data_vlass = hdu_vlass.data[0,0,:,:]
                   #print(hdu_vlass.data.shape)
                   img_vlass = hdu_vlass.data[0,0,:,:]
                   w1_vlass=wcs.WCS(hdu_vlass.header, naxis=2)
                   temp_vlass = fits.PrimaryHDU(data=data_vlass, header=w1_vlass.to_header())

                   #IR or Optical image
                   sdss_fits_f = '%s/SDSS_%s_r.fits' % (SDSS_dir, objns[m])
                   f=aplpy.FITSFigure(sdss_fits_f, slices=[0], north=True)
                   hdu_optical= fits.open(sdss_fits_f)[0]
                   data_optical = hdu_optical.data
                   f.show_colorscale(vmin=0.0, smooth=3, cmap='gray_r') #vmin=0.0, smooth=3 #cubehelix.cmap(reverse=True))
                   
                   rms =0.9* np.median(abs(img-np.median(img)))
                   print('rms first ->', rms)
                   #Plot the image in contours: 3 or 5
                   levs_positive = 5*rms*np.array([1,np.sqrt(2),2,np.sqrt(2)*2,4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64,64*np.sqrt(2),128,128*np.sqrt(2),256,256*np.sqrt(2)])
                   levs_negative = 5*rms*np.array([-1])
               
                   rms_nvss =0.9* np.median(abs(img_nvss-np.median(img_nvss)))
                   #Plot the image in contours: 3 or 5
                   levs_positive_nvss = 3*rms_nvss*np.array([1,np.sqrt(2),2,np.sqrt(2)*2,4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64,64*np.sqrt(2),128,128*np.sqrt(2),256,256*np.sqrt(2)])
                   levs_negative_nvss = 3*rms_nvss*np.array([-1])
                   
                   rms_vlass =0.9* np.median(abs(img_vlass-np.median(img_vlass)))
                   print(rms_vlass)
                   #Plot the image in contours: 3 or 5
                   levs_positive_vlass = 5*rms_vlass*np.array([1,2,4,8,16,32,64])#np.array([1,np.sqrt(2),2,np.sqrt(2)*2,4,4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2),32,32*np.sqrt(2),64,64*np.sqrt(2),128,128*np.sqrt(2),256,256*np.sqrt(2)])
                   levs_negative_vlass = 5*rms_vlass*np.array([-1])
 
                   f.show_contour(temp,dimensions=[0,1],colors='red',zorder=5,levels=levs_positive,slices=[0])#, alpha=0.3)
                   f.show_contour(temp_vlass,dimensions=[0,1],colors='blue',zorder=5,levels=levs_positive_vlass,slices=[0])#, alpha=0.3)
                   f.show_contour(temp_nvss,dimensions=[0,1],colors='orange',zorder=5,levels=levs_positive_nvss,slices=[0]) #'coral'
                   #bounding box
                   boxs = bboxs[m]
                   x1 = float(boxs.split('-')[0])
                   y1 = float(boxs.split('-')[1])
                   x2 = float(boxs.split('-')[2])
                   y2 = float(boxs.split('-')[3])
                   width = x2 - x1
                   height = y2 - y1
                   
                   radius = (max(width, height)/2.0 + 10) * pix_scale #deg
                   centre_ra = ras[m]
                   centre_dec = decs[m]
                   print(radius) 
                   f.recenter(centre_ra, centre_dec, radius=radius)
                   
                   
                   host_ra = sdss_ras[m]
                   host_dec = sdss_decs[m]
                   host_icrs = SkyCoord(ra=host_ra*u.degree, dec=host_dec*u.degree, frame='icrs') 
                   host_fk5 = host_icrs.transform_to('fk5')
                   f.show_markers(host_fk5.ra.value, host_fk5.dec.value, edgecolor='g', facecolor='g',
                                   marker='x', s=150, zorder=6)#, alpha=-0.5)
                   
                   source_name = sdss_names[m].split(' ')[1][0:5] + sdss_names[m].split(' ')[1][10:15]
                   print('SDSS name ->', source_name)
                   f.set_title(r'%s$%s$%s' % (source_name[0:5], source_name[5], source_name[6:11]), size=24)#fontname = 'serif') 
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
                else:
                   print('VLASS image not found!')





