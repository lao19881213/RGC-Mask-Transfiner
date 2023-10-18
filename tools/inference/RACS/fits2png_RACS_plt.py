import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pdb
from astropy.io import fits
import astropy.wcs as wcs
from PIL import Image
import matplotlib as mpl
import numpy as np
import argparse

def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', dest='inpdir', type=str, default='./', help='pred input png file directory')
parser.add_argument('--outdir', dest='outdir', type=str, default='./', help='output png file directory')
#parser.add_argument('--snr', dest='snr', type=int, default=200, help='SNR level')
#parser.add_argument('--rms', dest='rms', type=float, default=0.0, help='RMS value')
args = parser.parse_args()


def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

input_dir = args.inpdir
file_nms = os.listdir(input_dir)
for fn in file_nms:
    if not fn.endswith('.fits'):
       continue
   
    fits_file = fn
    hdu = fits.open(os.path.join(input_dir, fits_file))[0]
    if len(hdu.data.shape)==4:
       img_data = hdu.data[0,0,:,:]
    else :
       img_data = hdu.data

    if ((np.isnan(img_data)).all()):
       print("Image data %s all NaN" % fits_file)
    else:
       w = wcs.WCS(hdu.header).celestial
       ax=plt.subplot(projection=w)
       #fig = plt.figure()
       outdir = args.outdir
       pngfile = os.path.splitext(fn)[0] + ".png"
       #img_rms = args.rms #0.0000214 #0.0000296
       #img_std = np.std(hdu.data)
       #snr = args.snr
       #RACS -0.0005 (-0.5 mJy) to 0.005 (5 mJy) or -0.02 (-20 mJy) to 0.02 (20 mJy)
       plt.imsave(os.path.join(outdir, pngfile), img_data, vmin=-0.0005, vmax=0.005, origin='lower', cmap=CoolColormap()) # , cmap='Blues')
       #cmap=CoolColormap())
       print("Successful generate %s" % fn)


