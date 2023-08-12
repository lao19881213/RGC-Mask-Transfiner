import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u
import pandas as pd
import numpy as np
import matplotlib as mpl
import os
import argparse
import pycocotools.mask as cocomask
from astropy.io import fits
import astropy.wcs as wcs
import cv2
from photutils.datasets import make_4gaussians_image
from photutils.centroids import (centroid_1dg, centroid_2dg,
                                 centroid_com, centroid_quadratic)

from photutils.centroids import centroid_sources

parser = argparse.ArgumentParser()
parser.add_argument('--recsv', dest='recsv', type=str, default='', help='pred results file')
#parser.add_argument('--clsn', dest='clsn', type=str, default='FRII', help='class name')
parser.add_argument('--outdir', dest='outdir', type=str, default='./', help='output directory')
args = parser.parse_args()



def find_bbox_flux(bbox, fitsfile):
    hdu = fits.open(fitsfile)[0]

    # Set any NaN areas to zero or the interpolation will fail
    hdu.data[np.isnan(hdu.data)] = 0.0

    # Get vital stats of the fits file
    bmaj = hdu.header["BMAJ"]
    bmin = hdu.header["BMIN"]
    bpa = hdu.header["BPA"]
    xmax = hdu.header["NAXIS1"]
    ymax = hdu.header["NAXIS2"]
    try:
        pix2deg = hdu.header["CD2_2"]
    except KeyError:
        pix2deg = hdu.header["CDELT2"]
    # Montaged images use PC instead of CD
    if pix2deg == 1.0:
        pix2deg = hdu.header["PC2_2"]
    beamvolume = (1.1331 * bmaj * bmin)
    x1 = float(bbox.split('-')[0])
    y1 = float(bbox.split('-')[1])
    x2 = float(bbox.split('-')[2])
    y2 = float(bbox.split('-')[3])

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    int_flux = np.sum(box_data) #Jy/pix
    int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
    return int_flux


hetu_csv = pd.read_csv(args.recsv)

ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
boxs = hetu_csv['box'].values
labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values
pngs = hetu_csv['image_filename'].values
masks = hetu_csv['mask'].values

fits_dir = '/home/data0/lbq/inference_data/FIRST_fits'
bkg_dir = '/home/data0/lbq/inference_data/FIRST_rms'

boxs_new = []
centroid_ra = []
centroid_dec = []

for m in range(len(ras)):
      FIRST_fits = os.path.join(fits_dir, os.path.splitext(pngs[m])[0] + '.fits') 
      total_flux = find_bbox_flux(boxs[m], FIRST_fits)
      bkg_fits = os.path.join(bkg_dir, os.path.splitext(pngs[m])[0] + '_bkg.fits')
      bkg_flux = find_bbox_flux(boxs[m], bkg_fits)
      final_flux = total_flux - bkg_flux
      hetu_csv.loc[m,'int_flux'] = final_flux

      image = cv2.imread(os.path.join(fits_dir, pngs[m]))
      (height, width) = image.shape[:2]
      segm = {
                "size": [width, height],
                "counts": masks[m]}
      mask = cocomask.decode(segm)

      #method 1
      #cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
      #    cv2.CHAIN_APPROX_SIMPLE)
      #cnts = imutils.grab_contours(cnts)
      ##method 1
      #M = cv2.moments(cnts[0])
      #cX = int(M["m10"] / M["m00"])
      #cY = int(M["m01"] / M["m00"])
      
      hdu = fits.open(FIRST_fits)[0]
      #box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
      if len(hdu.data.shape)==4:
         data = hdu.data[0,0,:,:]
      else :
         data = hdu.data
      w1=wcs.WCS(hdu.header, naxis=2)
      #ra, dec = w1.wcs_pix2world([[cX, 132-cY]], 0)[0]
      

      
      #fig = plt.figure()
      #ax = plt.subplot(projection=w1)
      #ax.set_xlim([0, data.shape[1]])
      #ax.set_ylim([data.shape[0], 0])
      #ax.set_axis_off()

      img_x, img_y = w1.wcs_world2pix([[ras[m],decs[m]]],0).transpose()
      #ax.imshow(data, vmin=0, vmax=0.01, cmap='gist_heat', origin='lower')
      #ax.scatter(ra, dec, transform=ax.get_transform('fk5'), linewidths=1, marker="x", s=25,
      #     color='w')
      #ax.scatter(ras[m], decs[m], transform=ax.get_transform('fk5'), linewidths=1, marker="x", s=25,
      #     color='y')
      #method 2
      hdu_bkg = fits.open(bkg_fits)[0] 
      photutils_x, photutils_y = centroid_sources(data - hdu_bkg.data, img_x, img_y, mask=mask,
                        centroid_func=centroid_com)
      
      #photutils_x, photutils_y = centroid_quadratic(data, img_x, img_y, mask=mask)
      
      photutils_ra, photutils_dec = w1.wcs_pix2world([[photutils_x[0], photutils_y[0]]], 0)[0]
      #ax.scatter(photutils_ra, photutils_dec, transform=ax.get_transform('fk5'), linewidths=1, marker="x", s=25,
      #     color='g')
      if photutils_ra < 0 :
         photutils_ra = photutils_ra + 360
      centroid_ra.append(photutils_ra)
      centroid_dec.append(photutils_dec)

hetu_csv['centroid_ra'] = centroid_ra
hetu_csv['centroid_dec'] = centroid_dec
hetu_csv.to_csv(os.path.splitext(args.recsv)[0] + '_flux_centroid_fixed.csv', index = False)



   
