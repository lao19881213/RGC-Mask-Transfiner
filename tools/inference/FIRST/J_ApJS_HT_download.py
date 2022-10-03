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
from mpi4py import MPI
import linecache
import numpy as np
import requests
import time
from requests.packages.urllib3.exceptions import ReadTimeoutError, IncompleteRead
from requests.exceptions import ReadTimeout, SSLError, ConnectionError, ChunkedEncodingError
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--catalogfile', dest='catalogfile', type=str, default='J_ApJS_259_31_table1.dat.csv', help='catalog file')
parser.add_argument('--outdir', dest='outdir', type=str, default='/p9550/MWA/GLEAM/blao/hetu_images/deep_learn/v3_mask_results/HT', help='output fits file directory')
args = parser.parse_args()


# FIRST web form address

address='http://third.ucllnl.org/cgi-bin/firstcutout'
method='GET'

# fixed parameters

Equinox='J2000'
ImageSize='3.96' #in arcmin. this is the default value from the webpage
ImageType='FITS File'
MaxInt='10' #in mJy. this is the default value from the webpage

comm=MPI.COMM_WORLD
num_process=comm.Get_size()
rank=comm.Get_rank()


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
objn = re_csv['Name'].values
RAhs = re_csv['RAh'].values
RAms = re_csv['RAm'].values
RAss = re_csv['RAs'].values

DEsigs = re_csv['DE-'].values
DECds = re_csv['DEd'].values
DECms = re_csv['DEm'].values
DECss = re_csv['DEs'].values

pro_arr = np.array_split(range(len(objn)),num_process)
for n in pro_arr[rank]:
    RA= '%02d' % RAhs[n] + ' ' + '%02d' % RAms[n] + ' ' + '%.2f' % RAss[n] 
    Dec= DEsigs[n] + '%02d' % DECds[n] +' ' + '%02d' % DECms[n] +' '+ '%.2f' % DECss[n]

    ObjName = "J%02d%02d%.2f%s%02d%02d%.2f" % (RAhs[n], RAms[n], RAss[n], DEsigs[n], DECds[n], DECms[n], DECss[n])
           
    print ('ObjName is : ',ObjName)
    print ('RA is......: ',RA)
    print ('Dec is.....: ',Dec)

    out_dir = args.outdir
    outfile=os.path.join(out_dir, ObjName+'.fits')
    parameters = {'Equinox': Equinox, 'RA': RA, 'Dec': Dec, 'ImageSize': ImageSize, 'ImageType': ImageType, 'MaxInt': MaxInt}
    print(parameters)
    try:
        res = requests.post('https://third.ucllnl.org/cgi-bin/firstcutout', data=parameters, stream=False, proxies={"https": "https://192.168.6.12:3128"}, timeout=200)
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
