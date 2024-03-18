from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u
import pandas as pd
import os
import numpy as np

FIRST_fits_dir = '/home/data0/lbq/inference_data/FIRST_fits'

hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass_rm_lower.csv')

ras = hetu_csv['ra'].values
decs = hetu_csv['dec'].values
boxs = hetu_csv['box'].values
source_names = hetu_csv['source_name'].values
pngs = hetu_csv['image_filename'].values

for m in range(len(ras)):
    imagefilename = pngs[m]
    fits_file = os.path.splitext(imagefilename)[0] + ".fits" #fn.split('.')[0] + ".fits"
    hdu = fits.open(os.path.join(FIRST_fits_dir, fits_file))[0]
    #logger.info(os.path.join(input_dir, fits_file))
    w = wcs.WCS(hdu.header, naxis=2)
    
    x1 = float(boxs[m].split('-')[0])
    y1 = float(boxs[m].split('-')[1])
    x2 = float(boxs[m].split('-')[2])
    y2 = float(boxs[m].split('-')[3])


    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
 
    box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    peak_flux = np.nanmax(box_data)
    #logger.info('%d, %d, %d, %d' % (hdu.data.shape[0]-y2, hdu.data.shape[0]-y1, x1,x2))
    peak_xy_offset = np.where(box_data==np.nanmax(box_data))
    peak_x = x1 + peak_xy_offset[1][0]
    peak_y = hdu.data.shape[0]-y2 + peak_xy_offset[0][0]
    peak_ra, peak_dec = w.wcs_pix2world([[peak_x, peak_y]], 0)[0][0:2]
    if peak_ra < 0 :  #
       peak_ra = peak_ra + 360
   
    print(source_names[m], peak_ra, peak_dec)


    hetu_csv.loc[m,'ra'] = peak_ra
    hetu_csv.loc[m,'dec'] = peak_dec

hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass_rm_lower_peak_fixed.csv', index = False) 
