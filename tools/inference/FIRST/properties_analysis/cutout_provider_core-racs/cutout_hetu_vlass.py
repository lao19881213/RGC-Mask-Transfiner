import argparse
import numpy as np
import os
import pandas as pd
import math

parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
parser.add_argument('--surveys', default='0', type=str, help='surveys id, formats: 0,1,...')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
boxs = hetu_data['box'].values


clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}
surveys = {'0': 'FIRST', 
           '1': 'VLASS', 
           '2': 'RACS', 
           '3': 'GLEAM(f1,f2,f3,f4)', 
           '4': 'NVSS', 
           '5': 'WISE(w1,w2,w3,w4)', 
           '6': 'PanSTARRS[g,r,i,z,y]', 
           '7': 'SDSS[g,r,i]'}

surveys_download = []
for ss in args.surveys.split(','):
    surveys_download.append(surveys[ss])

surveys_final = ','.join(surveys_download)

#print(surveys_final)

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
          r = 2 #(np.max([xw, yw]) + 2) *1.8/60.0/2.0 #arcmin 
          cmd = 'python fetch_cutouts.py fetch -c %f,%f -s %s -r %d -g MOSAIC -o %s/%s --overwrite' \
                % (RA, DEC, surveys_final, math.ceil(r), args.outdir, clns[cln])    
          print(cmd)
          os.system(cmd) 
