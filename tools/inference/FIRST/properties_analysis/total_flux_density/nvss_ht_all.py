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
import json
import matplotlib.ticker as mticker

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
    #print(hdu.data.shape)
    box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    int_flux = np.sum(box_data) #Jy/pix
    int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
    return int_flux


hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv')

#ras = hetu_csv['centre_ra'].values
#decs = hetu_csv['centre_dec'].values
#boxs = hetu_csv['box'].values
#labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values
#FIRST_int_flux = hetu_csv['int_flux'].values

NVSS_bkg_dir = '/home/data0/lbq/inference_data/HT/NVSS_bkg'


NVSS_fits_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/NVSS'
NVSS_json_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/NVSS4flux_final_all'

#fitsfs = os.listdir(NVSS_fits_dir)

NVSS_int_flux_s = []
NVSS_boxs = []
for m in range(len(source_names)):
      print('Processing %s' % source_names[m])
      source_name = source_names[m] #os.path.splitext(source_names[m])[0].split('_')[1] 
      
      NVSS_fits = '%s/NVSS_%s.fits' % (NVSS_fits_dir, source_name)

      json_file = '%s/NVSS_%s.json' % (NVSS_json_dir, source_name)
      with open(json_file,'r', encoding='utf-8')as f:
           json_data = json.load(f)  
           obj = json_data['shapes'][0] 
           x1 = obj['points'][0][0]
           y1 = obj['points'][0][1]
           x2 = obj['points'][1][0]
           y2 = obj['points'][1][0]
           box = '{:.5f}-{:.5f}-{:.5f}-{:.5f}'.format(float(x1), float(y1), float(x2), float(y2))

      total_flux = find_bbox_flux(box, NVSS_fits)
      bkg_fits = os.path.join(NVSS_bkg_dir, 'NVSS_%s_bkg.fits' % source_name)
      bkg_flux = find_bbox_flux(box, bkg_fits)
      final_flux = total_flux - bkg_flux
      NVSS_int_flux_s.append(final_flux)
      NVSS_boxs.append(box)

hetu_csv['nvss_flux'] = NVSS_int_flux_s
hetu_csv['nvss_box'] = NVSS_boxs
hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv', index = False)
