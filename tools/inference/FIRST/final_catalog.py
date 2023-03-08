#!/usr/bin/env python


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
import os

import random

data_dir = "/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results"

match=pd.read_csv(os.path.join(data_dir,'FIRST_infer_part0-4_th0.1_first_14dec17_matched_cs_nocentre_final_paper.csv'),sep=',')

hetu=pd.read_csv(os.path.join(data_dir,'FIRST_HeTu.csv'),sep=',')

source_m = match['objectname'].values
bbox_m = match['box'].values
source = hetu['source_name'].values
bbox= hetu['box'].values

num=0
for n in range(match.shape[0]):
    for m in range(hetu.shape[0]):
       if source_m[n] == source[m] and bbox_m[n]==bbox[m]: 
          hetu.loc[m,'peak_flux'] = match['peak_flux'][n]
          num=num+1
          print(num)     

hetu.to_csv(os.path.join(data_dir,'FIRST_HeTu_paper.csv'), index=False)

