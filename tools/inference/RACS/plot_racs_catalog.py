import os
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
import argparse
import pandas as pd
from matplotlib.patches import Ellipse

parser = argparse.ArgumentParser()
parser.add_argument('--racscsv', help='RACS catalog csv file')
parser.add_argument('--inpdir', help='pred input png file directory')
parser.add_argument('--outdir', help='pred output png file directory')
#parser.add_argument('--rms', type=float, help='rms noise')
args = parser.parse_args()

result_file = args.racscsv 
def CoolColormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#000069', '#00188a', '#0d6bff', '#1abaff',
                                                                 '#d9ffff', '#ffffff'], 256)

input_dir = args.inpdir 
file_nms = os.listdir(input_dir)

racs_csv = pd.read_csv(result_file)

ra_sl = racs_csv['ra'].values
dec_sl = racs_csv['dec'].values

#snr = 40
#rms = args.rms

#pro_arr = np.array_split(np.arange(len(file_nms)),num_process)
for n in range(len(file_nms)):
    if not file_nms[n].endswith('.png'):
       continue
    
    fits_file = os.path.splitext(file_nms[n])[0] + ".fits"
    hdu = fits.open(os.path.join(input_dir, fits_file))
    image = hdu[0].data
    h = hdu[0].header
    w = wcs.WCS(hdu[0].header, naxis=2)
    width = hdu[0].data.shape[1]
    height = hdu[0].data.shape[0]

    bottom_left = [0, 0]
    top_left = [0, height - 1]
    top_right = [width - 1, height - 1]
    bottom_right = [width - 1, 0]

    ret = np.zeros([4, 2])
    ret[0, :] = w.wcs_pix2world([bottom_left], 0)[0][0:2]
    ret[1, :] = w.wcs_pix2world([top_left], 0)[0][0:2]
    ret[2, :] = w.wcs_pix2world([top_right], 0)[0][0:2]
    ret[3, :] = w.wcs_pix2world([bottom_right], 0)[0][0:2]
    RA_min, DEC_min, RA_max, DEC_max = np.min(ret[:, 0]),   np.min(ret[:, 1]),  np.max(ret[:, 0]),  np.max(ret[:, 1])

    #ax = plt.subplot(projection=w)
    ax = plt.subplot()
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    ax.set_axis_off()
    plt.imshow(np.flipud(image), origin='lower', vmin=-0.0005, vmax=0.02, cmap='Blues')
    k= 15
    wb = (k,0,image.shape[1]-15,k)
    #print("bmaj %f" % h['bmaj'])
    pix = abs(h['CDELT1'])
    b = (wb[0]-1.5*h['bmaj']/pix,wb[2]+1.5*h['bmaj']/pix,h['bmaj']/pix,h['bmin']/pix,h['bpa'])
    e = Ellipse(xy=(wb[1]+b[2],wb[2]+b[3]*(1.2)),width=b[2],height=b[3],angle=90-b[4],ec='k',facecolor='grey')
    ax.add_artist(e)           
    for m in range(len(ra_sl)):
        ra_sl1 = float(ra_sl[m])
        dec_sl1 = float(dec_sl[m])
        if ra_sl1 <= RA_max and ra_sl1 >= RA_min and dec_sl1 <= DEC_max and dec_sl1 >= DEC_min:
           sl1_x, sl1_y = w.wcs_world2pix([[ra_sl1,dec_sl1]],0).transpose()
           ax.scatter(sl1_x, height-sl1_y, marker="s", s=50,
           edgecolor='y', facecolors='none')
           #ax.scatter(ra_sl1, dec_sl1, transform=ax.get_transform('fk5'), marker="s", s=100,
           #edgecolor='y', color='')

           plt.rcParams["font.family"] = "Times New Roman"
    outdir = args.outdir #"/o9000/MWA/GLEAM/G0008_images/deep_learn/vlass_test/split_img_2.1_png_pred"
    pngfile = file_nms[n] 
    plt.savefig(os.path.join(outdir, pngfile))
    #plt.savefig(os.path.join(outdir, pdffile))
    plt.clf()

