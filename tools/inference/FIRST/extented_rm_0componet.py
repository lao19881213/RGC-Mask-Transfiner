import os
import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
from PIL import Image
import argparse
import pandas as pd
import linecache

parser = argparse.ArgumentParser()
parser.add_argument('--incsv', help='input csv file')
parser.add_argument('--compcsv', help='componet csv file')
args = parser.parse_args()
#tags = []

csv_org = pd.read_csv(args.incsv)

csv_comp = pd.read_csv(args.compcsv)
index = []
for m in range(csv_comp.shape[0]):
    if int(csv_comp['comp_cnt'][m])==0:
       index.append(int(csv_comp['index'][m]))
       
       
df_new = csv_org.drop(index=index)
df_new.to_csv('%s_rm_0comp.csv' % os.path.splitext(args.incsv)[0], index=False) 


