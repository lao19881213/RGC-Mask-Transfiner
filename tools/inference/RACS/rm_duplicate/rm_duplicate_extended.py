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
parser.add_argument('--reptxt', help='repeat txt file')
parser.add_argument('--outdir', help='output results directory')
parser.add_argument('--outcsv', help='output csv file')
args = parser.parse_args()

fr12_csv = args.fr12csv

csv_fr12 = pd.read_csv(fr12_csv)

print(len(csv_fr12))
index_all = []
with open(args.reptxt) as f:
     for line in f:
         line_re = line.split('\n')[0]
         line_p = line_re.split(', ')
         #print(line_p)
         line_l = len(line_p)
         rep_row = line_p[5:line_l+1]
         print(rep_row)
         for i in range(len(rep_row)):
             if rep_row[i]!='':
                index = int(rep_row[i])
                print(index)
                index_all.append(index)

csv_fr12.drop(index_all, inplace=True)

csv_fr12.to_csv(os.path.join(args.outdir, args.outcsv), index=False) 
