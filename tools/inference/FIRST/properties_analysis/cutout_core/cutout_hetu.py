import argparse
import numpy as np
import os
import pandas as pd
import math

parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
boxs = hetu_data['box'].values


clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT'}
surveys = ['FIRST', 'VLASS', 'NVSS', 'WISE', 'PanSTARRS', 'SDSS']
nn=0
for m in range(len(labels)):
    for cln in clns.keys():
        #print(cls)
        if(labels[m]==cln):
          RA = float(ras[m])
          DEC = float(decs[m])
          x1 = float(boxs[m].split('-')[0])
          y1 = float(boxs[m].split('-')[1])
          x2 = float(boxs[m].split('-')[2])
          y2 = float(boxs[m].split('-')[3])
          xw = x2 - x1
          yw = y2 - y1
          r = (np.max([xw, yw]) + 2) *1.8/60.0/2.0 #arcmin 
          cmd = 'python fetch_cutouts.py fetch -c %f,%f -s %s -r %d -g MOSAIC -o %s/%s --overwrite' \
                % (RA, DEC, surveys[0], math.ceil(r), args.outdir, clns[cln])    
          print(cmd)
          os.system(cmd) 
