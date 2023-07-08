import pdb
import pandas as pd
from astropy import coordinates
from astropy.table import Table
from astropy import units, wcs
from astroquery.skyview import SkyView
import argparse
import numpy as np
import astropy.units as u
import os


parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
source_names = hetu_data['source_name'].values
boxs = hetu_data['box'].values

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

skyview = SkyView()

tags_non = []
for m in range(len(labels)):
        for cln in clns.keys():
            #print(cln)
            if(labels[m]==cln):
               fits_fn = 'NVSS_%s.fits' % (source_names[m])
               if os.path.isfile(f'%s/%s/%s/%s' % (args.outdir, clns[cln], 'NVSS', fits_fn)):
                   print('%s already exists!' % (fits_fn))
               else:
                   object_coord = coordinates.SkyCoord(ra=ras[m], dec=decs[m], unit=(u.deg, u.deg))#, frame='icrs')#frame='fk5')
                   fits_fn = 'NVSS_%s.fits' % (source_names[m])
                   try:
                      paths_nvss = SkyView.get_images(position=object_coord.to_string('hmsdms'),
                                      coordinates='J2000',
                                      survey='NVSS', width=3.96*u.arcmin)[0]
                      #print(links_gleam)
                      print("download %s ..." % source_names[m])
                      hdu = paths_nvss
                      hdu[0].header['BMAJ']=45.0/3600.0
                      hdu[0].header['BMIN']=45.0/3600.0
                      hdu[0].header['BPA']=0
                      hdu[0].header['RESTFREQ']=1.4e9
                      hdu.writeto("%s/%s/%s/%s" % (args.outdir, clns[cln], 'NVSS', fits_fn), overwrite=True)
                   except:
                      print('Not found ', source_names[m])#name)
                      tags_non.append("{},{},{},{},{}".format(m,labels[m],source_names[m],ras[m],decs[m]))
                                              

resultsData_non = tags_non 
with open(os.path.join('./', 'NOT_FOUND_nvss.txt'), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))
