import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import matplotlib as mpl
import numpy as np
import pandas as pd
#import argparse
#from mpi4py import MPI

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

ht_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_ht_flux_fixed_vlass.csv')
source_names = ht_csv['source_name'].values
ras = ht_csv['centre_ra'].values
decs = ht_csv['centre_dec'].values
boxs = ht_csv['box'].values
input_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis/HT/FIRST'
outdir = '/home/data0/lbq/inference_data/HT/FIRST'


def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

for n in range(len(source_names)):
    fits_file = '%s.fits' % source_names[n]
    fn = fits_file
    box = boxs[n]
    ra = ras[n]
    dec = decs[n]
    try:
       hdu = fits.open(os.path.join(input_dir, fits_file))[0]
    except:
       print("fits file %s is broken!" % fits_file)
       continue
       #sys.exit()

    if len(hdu.data.shape)==4:
       img_data = hdu.data[0,0,:,:]
    else :
       img_data = hdu.data
    w = wcs.WCS(hdu.header, naxis=2)
    x1 = float(box.split('-')[0])
    y1 = float(box.split('-')[1])
    x2 = float(box.split('-')[2])
    y2 = float(box.split('-')[3])
    width = x2 - x1
    height = y2 - y1
    centre_x, centre_y = w.wcs_world2pix([[ra,dec]],0).transpose()
    x1_new = centre_x - width/2.0
    x2_new = centre_x + width/2.0
    y1_new = centre_y - height/2.0
    y2_new = centre_y + height/2.0

    top, left, bottom, right = img_data.shape[0] - int(y2_new), int(x1_new), img_data.shape[0] - int(y1_new), int(x2_new)
    #fig = plt.figure()
    ax=plt.subplot(projection=w)
    ax.set_xlim([0, img_data.shape[1]])
    ax.set_ylim([img_data.shape[0], 0])
    ax.set_axis_off()
    pngfile = os.path.splitext(fn)[0] + ".png"
    datamax = np.nanmax(img_data)
    datamin = np.nanmin(img_data)
    if np.isnan(datamax) and np.isnan(datamin):
       plt.imshow(img_data, cmap=CoolColormap(),origin='lower')
       print(fits_file + " is all nan, skip it")
    else:
       datawide = datamax - datamin
       image_data = np.log10((hdu.data - datamin) / datawide * 1000 + 1) / 3 
       plt.imshow(image_data, cmap=CoolColormap(),origin='lower')

    ax.add_patch(
    plt.Rectangle((left, top),
               abs(left - right),
	       abs(top - bottom), fill=False,
	       edgecolor='red', linewidth=2)
               )
    plt.savefig(os.path.join(outdir, pngfile))			  
    print("Successful generate %s" % fn)
    plt.clf()
