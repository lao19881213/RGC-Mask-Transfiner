#! /usr/local/bin/python

import string
import os
import argparse
import linecache
import numpy as np
import time
import pandas as pd


CatalogFile = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/J_ApJS_259_31_dat.csv'


re_csv = pd.read_csv(CatalogFile)
objn = re_csv['Name'].values
RAhs = re_csv['RAh'].values
RAms = re_csv['RAm'].values
RAss = re_csv['RAs'].values

DEsigs = re_csv['DE-'].values
DECds = re_csv['DEd'].values
DECms = re_csv['DEm'].values
DECss = re_csv['DEs'].values

tags = []
#pro_arr = np.array_split(range(len(objn)),num_process)
for n in range(len(objn)):

    ObjName = "J%02d%02d%.2f%s%02d%02d%.2f" % (RAhs[n], RAms[n], RAss[n], DEsigs[n], DECds[n], DECms[n], DECss[n])
    RA = RAhs[n]*15.0 + (RAms[n]/60.0)*15.0 + (RAss[n]/3600.0)*15.0
    Dec = DECds[n] + DECms[n]/60.0 + DECss[n]/3600.0
    if DEsigs[n] == '-':
       Dec = -Dec
    else:
       Dec = Dec     
    print ('ObjName is : ',ObjName)
    print ('RA is......: ',RA)
    print ('Dec is.....: ',Dec)
    tags.append('%s,%f,%f' % (ObjName, RA, Dec))

with open(os.path.join('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results', 'J_ApJS_HT_catalog.csv'), 'w') as f:
     f.write(os.linesep.join(tags))

