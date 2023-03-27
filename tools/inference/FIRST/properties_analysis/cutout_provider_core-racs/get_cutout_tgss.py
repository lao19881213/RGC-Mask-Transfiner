# script to obtain cutouts around using and RA and DEC list
# Aayush Saxena
# Modified by Baoqiang Lao

import pyvo as vo
import subprocess
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
args = parser.parse_args()

#######################################
# Input text file should have RA DEC
#######################################
hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
boxs = hetu_data['box'].values
names = hetu_data['source_name'].values

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

vo.__file__
url='https://vo.astron.nl/tgssadr/q_fits/imgs/siap.xml'


for m in range(len(labels)):
    for cln in clns.keys():
        #print(cls)
        if(labels[m]==cln):
           os.chdir('%s/%s/TGSS' % (args.outdir, clns[cln]))
           query = vo.sia.SIAQuery(url)
           query.format = 'image/fits'
           
           #######################################
           # Choose the cutout size (degrees)
           query.size = 0.1
           #######################################
           
           name=names[m]
           ra=ras[m]
           dec=decs[m]
           #print(float(ra), float(dec)) 
           query.pos = (float(ra), float(dec))
           
           results = query.execute()
           #print(results)
           for image in results:
               print ("Downloading %s..." %name)
               image.cachedataset(filename="TGSS_%s.fits" %name)

print("All done!")
