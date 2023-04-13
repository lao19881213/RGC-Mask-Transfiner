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

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)


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


hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv')

ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
boxs = hetu_csv['box'].values
labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

root_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis'

boxs_new = []
for m in range(len(ras)):
      for cln in clns.keys():
          #print(cln)
          if(labels[m]==cln):
             FIRST_fits = '%s/%s/FIRST/%s.fits' % (root_dir, clns[labels[m]], source_names[m])
             hdu_FIRST = fits.open(FIRST_fits)[0] 
             try:
                 pix_scale_hetu = hdu_FIRST.header["CD2_2"]
             except KeyError:
                 pix_scale_hetu = hdu_FIRST.header["CDELT2"]
             
             TGSS_fits = '%s/%s/TGSS/%s.FITS' % (root_dir, clns[labels[m]], source_names[m])
             try:
                 hdu_TGSS = fits.open(TGSS_fits)[0]
             except:
                 continue
             try:
                 pix_scale_tgss = hdu_TGSS.header["CD2_2"]
             except KeyError:
                 pix_scale_tgss = hdu_TGSS.header["CDELT2"]

             x1 = float(boxs[m].split('-')[0])
             y1 = float(boxs[m].split('-')[1])
             x2 = float(boxs[m].split('-')[2])
             y2 = float(boxs[m].split('-')[3])
             width = x2 - x1
             height = y2 - y1
             factor = pix_scale_tgss / pix_scale_hetu
             w = wcs.WCS(hdu_TGSS.header, naxis=2)
             centre_x, centre_y = w.wcs_world2pix([[ras[m],decs[m]]],0).transpose() 
             width_new = width / factor
             height_new = height / factor
             x1_new = centre_x - width/2.0
             x2_new = centre_x + width/2.0
             y1_new = centre_y - height/2.0
             y2_new = centre_y + height/2.0 
             
             out_dir = '%s/%s/TGSS_png' % (root_dir, clns[labels[m]])
             if len(hdu_TGSS.data.shape)==4:
                img_data = hdu_TGSS.data[0,0,:,:]
             else :
                img_data = hdu_TGSS.data
             #w = wcs.WCS(hdu_TGSS.header, naxis=2)
             fig = plt.figure()
             ax = plt.subplot(projection=w)
             ax.set_xlim([0, img_data.shape[1]])
             ax.set_ylim([img_data.shape[0], 0])
             ax.set_axis_off()
             datamax = np.nanmax(img_data)
             datamin = np.nanmin(img_data)
             datawide = datamax - datamin
             image_data = np.log10((img_data - datamin) / datawide * 1000 + 1) / 3

             #img_rms = 0.00002
             ax.imshow(image_data, origin='lower', cmap=CoolColormap()) 
             #boxs_new.append('%s,%.5f-%.5f-%.5f-%.5f' % (source_names[m], x1_new, y1_new, x2_new, y2_new))
             top, left, bottom, right = hdu_TGSS.data.shape[0]-y2_new, x1_new, hdu_TGSS.data.shape[0]-y1_new, x2_new
             ax.add_patch(
             plt.Rectangle((left, top),abs(left - right),abs(top - bottom), \
                          fill=False, edgecolor='r', linewidth=2)
                         )
             plt.savefig(os.path.join(out_dir, '%s.png' % source_names[m]))
             plt.clf()

   
