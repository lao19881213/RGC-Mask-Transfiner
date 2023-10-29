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
from astropy.io import fits
import astropy.wcs as wcs
import cv2

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)


parser = argparse.ArgumentParser()
parser.add_argument('--recsv', dest='recsv', type=str, default='', help='pred results file')
#parser.add_argument('--clsn', dest='clsn', type=str, default='FRII', help='class name')
parser.add_argument('--outdir', dest='outdir', type=str, default='./', help='output directory')
args = parser.parse_args()



def find_bbox_flux(bbox, fitsfile):
    hdu = fits.open(fitsfile)[0]
    print('Processing %s ...' % fitsfile)
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
    #test
    #w = wcs.WCS(hdu.header, naxis=2)
    #fig = plt.figure()
    #ax = plt.subplot(projection=w)
    #ax.set_xlim([0, box_data.shape[1]])
    #ax.set_ylim([box_data.shape[0], 0])
    #ax.set_axis_off()
    #datamax = np.nanmax(box_data)
    #datamin = np.nanmin(box_data)
    #datawide = datamax - datamin
    #image_data = np.log10((box_data - datamin) / datawide * 1000 + 1) / 3 
    #ax.imshow(image_data, origin='lower', cmap=CoolColormap())
    #pngf = 'J' + fitsfile.split('J')[1].replace('fits', 'png') 
    #plt.savefig(os.path.join('/home/data0/lbq/inference_data/FIRST_flux_test/', pngf))
    #print(os.path.join('/home/data0/lbq/inference_data/FIRST_flux_test/', pngf))
    #plt.clf()
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
      
hetu_csv.to_csv(os.path.splitext(args.recsv)[0] + '_flux_fixed.csv', index = False)



   