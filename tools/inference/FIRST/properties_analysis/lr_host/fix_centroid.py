from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u
import pandas as pd
import numpy as np
import os
import pycocotools.mask as cocomask
from astropy.io import fits
import astropy.wcs as wcs
import cv2
import imutils

hetu_csv = pd.read_csv('/media/hero/Intel6/RACS_mid/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass_rm_lower_peak_fixed.csv')

ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
boxs = hetu_csv['box'].values
labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values
pngs = hetu_csv['image_filename'].values
masks = hetu_csv['mask'].values

fits_dir = '/media/hero/Intel6/RACS_mid/FIRST_fits'

boxs_new = []
centroid_ra = []
centroid_dec = []

for m in range(len(ras)):
      FIRST_fits = os.path.join(fits_dir, os.path.splitext(pngs[m])[0] + '.fits') 

      image = cv2.imread(os.path.join(fits_dir, pngs[m]))
      (height, width) = image.shape[:2]
      segm = {
                "size": [width, height],
                "counts": masks[m]}
      mask = cocomask.decode(segm)

      #method 1
      cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)
      cnts = imutils.grab_contours(cnts)
      #method 1
      M = cv2.moments(cnts[0])
      if M["m00"] == 0.0:
         centroid_ra.append(ras[m])
         centroid_dec.append(decs[m])
      else: 
         cX = int(M["m10"] / M["m00"])
         cY = int(M["m01"] / M["m00"])
         
         hdu = fits.open(FIRST_fits)[0]
         #box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
         if len(hdu.data.shape)==4:
            data = hdu.data[0,0,:,:]
         else :
            data = hdu.data
         w1=wcs.WCS(hdu.header, naxis=2)
         ra, dec = w1.wcs_pix2world([[cX, 132-cY]], 0)[0]

         centroid_ra.append(ra)
         centroid_dec.append(dec)

hetu_csv['centroid_ra'] = centroid_ra
hetu_csv['centroid_dec'] = centroid_dec
hetu_csv.to_csv('/media/hero/Intel6/RACS_mid/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass_rm_lower_peak_centroid_fixed.csv', index = False)



   
