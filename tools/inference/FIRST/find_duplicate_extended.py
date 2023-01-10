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
import linecache

parser = argparse.ArgumentParser()
parser.add_argument('--fr12csv', help='fr12 csv file')
parser.add_argument('--inpdir', help='input fits and png file directory')
parser.add_argument('--outdir', help='output results directory')
parser.add_argument('--outfile', help='output file')
args = parser.parse_args()

input_dir = args.inpdir 

fr12_csv = args.fr12csv

csv_fr12 = pd.read_csv(fr12_csv)

ras = csv_fr12['centre_ra'].values

decs = csv_fr12['centre_dec'].values

boxs = csv_fr12['box'].values

image_files = csv_fr12['imagefilename'].values


tags = []

for n in range(len(ras)):
   rep_row = []
   fits_file = os.path.splitext(image_files[n])[0] + ".fits"
   hdu = fits.open(os.path.join(input_dir, fits_file))
   image = hdu[0].data
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

   x1 = float(boxs[n].split('-')[0])
   y1 = float(boxs[n].split('-')[1])
   x2 = float(boxs[n].split('-')[2])
   y2 = float(boxs[n].split('-')[3])

   x1 = int(x1)
   y1 = int(y1)
   x2 = int(x2)
   y2 = int(y2)
   print(x1,x2,y1,y2)
   y1_new = hdu[0].data.shape[0]-y2
   y2_new = hdu[0].data.shape[0]-y1
   
    
   for m in range(n+1,len(ras)):
       ra_first = float(ras[m])
       dec_first = float(decs[m])
       img_x, img_y = w.wcs_world2pix([[ra_first,dec_first]],0).transpose()
       if ra_first <= RA_max and ra_first >= RA_min and dec_first <= DEC_max and dec_first >= DEC_min:
          if img_x <= x2 and img_x >= x1 and img_y <= y2_new and img_y >= y1_new:
             rep_row.append(m) 
   str_row = ", ".join(str(i) for i in rep_row)
   tags.append('%d, %s, %s, %s, %s, %s' % (n, image_files[n], boxs[n], ras[n], decs[n], str_row))                 


with open(os.path.join(args.outdir, args.outfile), 'w') as f:
     f.write(os.linesep.join(tags))
