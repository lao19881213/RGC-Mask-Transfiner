#! /usr/local/bin/python

#===============================================================================
# To repeatedly fill in the NVSS postage stamp survey webpage at
#
#       http://third.ucllnl.org/cgi-bin/firstcutout
#
#
# Parameters required:
#
#   equinox        (set to "J2000")
#   image size     (set to "4.5" arcmin)
#   image type     (set to "FITS file")
#   max intensity  (set to "10" mJy)
#
# Prompts for a text file containing lines of:
#
# source hh mm ss.ss dd mm ss.s
#
# or 
# source hh:mm:ss.ss +dd:mm:ss.s
#
# (the source's name is used for the name of the outfile file (+.fits))
#-------------------------------------------------------------------------------
# Tamela Maciel 30 June 2014
# Modified by Baoqiang Lao 22 Jan 2022
#===============================================================================
import string
import os
import argparse
import linecache
import numpy as np
import requests
import time
from requests.packages.urllib3.exceptions import ReadTimeoutError, IncompleteRead
from requests.exceptions import ReadTimeout, SSLError, ConnectionError, ChunkedEncodingError
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--catalogfile', dest='catalogfile', type=str, default='/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results', help='catalog file')
parser.add_argument('--outdir', dest='outdir', type=str, default='/tmp', help='output fits file directory')
args = parser.parse_args()


# FIRST web form address

address='http://third.ucllnl.org/cgi-bin/firstcutout'
method='GET'

# fixed parameters

Equinox='J2000'
ImageSize='3.96' #in arcmin. this is the default value from the webpage
ImageType='FITS File'
MaxInt='10' #in mJy. this is the default value from the webpage


CatalogFile = args.catalogfile

print ('------------------------------------------------------')
print ('Extracting .fits files from FIRST cutout                    ')
print ('------------------------------------------------------')
print ('Using parameters:')
print ('  equinox.....: ',Equinox)
print ('  image size (arcmin).....: ',ImageSize)
print ('  image type........: ',ImageType)
print ('  max intensity (mJy).......: ',MaxInt)
print ('-----------------------------')
print ('  catalog file : '+CatalogFile)
print ('------------------------------------------------------')

re_csv = pd.read_csv(CatalogFile)
objn = re_csv['source_name'].values
labels = re_csv['label'].values

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

#pro_arr = np.array_split(range(len(objn)),num_process)
for m in range(len(objn)):#pro_arr[rank]:
      for cln in clns.keys():
          #print(cln)
          if(labels[m]==cln):
             items = objn[m]
             if(len(items.split('-'))>=2):
               print(items)
               RAlist=items.split('-')[0].split('J')[1]
               Declist=items.split('-')[1]
               RA=RAlist[0:2]+' '+RAlist[2:4]+' '+ RAlist[4:]
               Dec='-' + Declist[0:2]+' '+Declist[2:4]+' '+ Declist[4:]
             elif(len(items.split('+'))>=2):
               RAlist=items.split('+')[0].split('J')[1]
               Declist=items.split('+')[1]
               RA=RAlist[0:2]+' '+RAlist[2:4]+' '+RAlist[4:]
               Dec= Declist[0:2]+' '+Declist[2:4]+' '+Declist[4:]

             ObjName = items
                    
             print ('ObjName is : ',ObjName)
             print ('RA is......: ',RA)
             print ('Dec is.....: ',Dec)

             out_dir = args.outdir
             outfile=os.path.join('%s/%s/FIRST/%s.fits' % (out_dir, clns[cln], ObjName))
             if os.path.isfile(f'%s/%s/FIRST/%s.fits' % (out_dir, clns[cln], ObjName)):
                  print('%s/%s/FIRST/%s.fits already exists!' % (out_dir, clns[cln], ObjName))
             else: 
                  parameters = {'Equinox': Equinox, 'RA': RA, 'Dec': Dec, 'ImageSize': ImageSize, 'ImageType': ImageType, 'MaxInt': MaxInt}
                  print(parameters)
                  try:
                      res = requests.post('https://third.ucllnl.org/cgi-bin/firstcutout', data=parameters, stream=False, timeout=200)
                      with open(outfile, 'wb') as out:
                           for i in res:
                               out.write(i)
                  except (ReadTimeout, IncompleteRead, SSLError, ConnectionError, ChunkedEncodingError):
                      continue
                  #except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                  #    if 'bad handshake' in str(e) or '10054' in str(e):
                  #       continue
                  #    else:
                  #       raise Exception(e)
                  
                  #time.sleep(10)
                  print ('Source ',ObjName,' processed, output in '+ObjName+'.fits')

print ('')
print ('------------------------------------------------------')
print ('Done!')
print ('------------------------------------------------------')
