import string
import os
import argparse
import linecache
import numpy as np
import time
import pandas as pd
from astropy.coordinates import SkyCoord

parser = argparse.ArgumentParser()
parser.add_argument('--catalogfile', dest='catalogfile', type=str, default='/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results', help='catalog file')
parser.add_argument('--pngdir', dest='pngdir', type=str, default='/tmp', help='output fits file directory')
parser.add_argument('--cln', dest='classname', type=str, default='ht', help='class name')
args = parser.parse_args()

CatalogFile = args.catalogfile

re_csv = pd.read_csv(CatalogFile)
objn = re_csv['source_name'].values

cln = args.classname

vlass_pngns = os.listdir(args.pngdir)
tags = []
#pro_arr = np.array_split(range(len(objn)),num_process)
for m in range(len(objn)):#pro_arr[rank]:
    items = objn[m]
    #print(items)
    if(len(items.split('-'))>=2):
      RAlist=items.split('-')[0].split('J')[1]
      Declist=items.split('-')[1]
      ra_first = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
      dec_first = 0 - (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
    elif(len(items.split('+'))>=2):
      RAlist=items.split('+')[0].split('J')[1]
      Declist=items.split('+')[1]
      ra_first = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
      dec_first = (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
    print(ra_first, dec_first)
    #sky_first = SkyCoord.from_name(items)
    #ra_first = sky_first.ra.value
    #dec_first = sky_first.dec.value
    #print(ra_first, dec_first)
    for n in range(len(vlass_pngns)):
        pngn = vlass_pngns[n][0:19]
        # J072959.65+282815.6
        if(len(pngn.split('-'))>=2):
          RAlist=pngn.split('-')[0].split('J')[1]
          Declist=pngn.split('-')[1]
          ra_vlass = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
          dec_vlass = 0 - (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0)
        elif(len(pngn.split('+'))>=2):
          RAlist=pngn.split('+')[0].split('J')[1]
          Declist=pngn.split('+')[1]
          ra_vlass = (float(RAlist[0:2])*15 + float(RAlist[2:4])/60.0*15.0 + float(RAlist[4:])/3600.0*15.0)
          dec_vlass = (float(Declist[0:2]) + float(Declist[2:4])/60.0 + float(Declist[4:])/3600.0) 
        #sky_vlass = SkyCoord.from_name(pngn)
        #ra_vlass = sky_vlass.ra.value
        #dec_vlass = sky_vlass.dec.value
         
        if((abs((ra_first-ra_vlass)*3600.0) <=1.0) and (abs((dec_first-dec_vlass)*3600.0) <=1.0)):
           #print(pngn)
           #print((ra_first-ra_vlass)*3600.0, dec_first - dec_vlass)
           print('Cross matched FIRST and VLASS: %s, %s' % (items, pngn))
           break
        else:
           if n == len(vlass_pngns) - 1:
              tags.append("{},{}".format(m,objn[m])) 

with open(os.path.join('./', 'NOT_FOUND_VLASS_PNG_%s.txt' % cln), 'w') as fn:
     fn.write(os.linesep.join(tags))
            
