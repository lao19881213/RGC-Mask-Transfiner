import os, sys, math
import os.path as osp

import numpy as np

import os.path as osp

import argparse
import logging

from pyds9 import *

import linecache

# source /home/blao/rge_resnet_fpn/bashrc
# module load ds9/centos7-v8.3 
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', dest='inpdir', type=str, default='./', help='pred input png file directory')
parser.add_argument('--outdir', dest='outdir', type=str, default='./', help='output png file directory')
#parser.add_argument('--snr', dest='snr', type=int, default=200, help='SNR level')
#parser.add_argument('--rms', dest='rms', type=float, default=0.0, help='RMS value')
args = parser.parse_args()

fits_dir = args.inpdir
png_dir = args.outdir

print(ds9_targets())
d = DS9('7f000001:43271')

file_nms = os.listdir(fits_dir)
for fn in file_nms:
    if not fn.endswith('.fits'):
       continue

    png = fn.replace('.fits', '.png')
    if os.path.exists(osp.join(png_dir, png)):
       logger.info("%s was exists.\n"  % osp.join(png_dir, png))
    else:
       try:
          d.set("file "+ "%s" % os.path.join(fits_dir, fn))
          d.set('zoom to fit')
          d.set('scale mode minmax')
          d.set('scale log')
          d.set('cmap Cool')
          d.set('export %s' % osp.join(png_dir, png))
       except:
          continue 
