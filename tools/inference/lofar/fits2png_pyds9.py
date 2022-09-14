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
parser = argparse.ArgumentParser(description='fits2png')

parser.add_argument('--catalog_file', dest='catalog_file', type=str, default='Mingo.txt', help='catalog file')

args = parser.parse_args()

catalog_file = args.catalog_file

data_dir = "/p9550/LOFAR/LoTSS-DR1" #os.path.join(cur_dir, '..', 'data')
fits_dir = osp.join(data_dir, 'Mingo_fits')
png_dir = osp.join(data_dir, 'Mingo_png')

print(ds9_targets())
d = DS9()

with open('%s' % osp.join(data_dir, catalog_file), 'r') as file:
  print ('File opened...')
  print ('')
  linecount=len(file.readlines())

for n in range(linecount): 
    line = linecache.getline(osp.join(data_dir, catalog_file), n+1)

    fn = line.split('\n')[0]
    png = fn.replace('.fits', '.png')
    if os.path.exists(osp.join(png_dir, png)):
       logger.info("%s was exists.\n"  % osp.join(png_dir, png))
    else:
       d.set("file "+ "%s" % os.path.join(fits_dir, fn))
       d.set('zoom to fit')
       d.set('scale mode minmax')
       d.set('scale log')
       d.set('cmap Cool')
       d.set('export %s' % osp.join(png_dir, png)) 
