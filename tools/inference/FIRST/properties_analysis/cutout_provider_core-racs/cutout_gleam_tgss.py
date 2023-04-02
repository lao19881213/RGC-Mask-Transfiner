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
parser.add_argument('--surveys', default='0', type=str, help='surveys id, formats: 0,1,...')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
source_names = hetu_data['source_name'].values
boxs = hetu_data['box'].values

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}
surveys = {'0': 'GLEAM',
           '1': 'TGSS'}

skyview = SkyView()

tags_non = []
for m in range(len(labels)):
        for cln in clns.keys():
            #print(cln)
            if(labels[m]==cln):
               RA = float(ras[m])
               DEC = float(decs[m])
               x1 = float(boxs[m].split('-')[0])
               y1 = float(boxs[m].split('-')[1])
               x2 = float(boxs[m].split('-')[2])
               y2 = float(boxs[m].split('-')[3])
               xw = x2 - x1
               yw = y2 - y1
               r = (np.max([xw, yw]) + 2) *1.8/2.0 #arcmin
               radius = r * u.arcsec
               object_coord = coordinates.SkyCoord(ra=ras[m], dec=decs[m], unit=(u.deg, u.deg))#, frame='icrs')#frame='fk5')
               band="170-231 MHz"
               fits_fn = '%s_%s.fits' % (surveys[args.surveys], source_names[m])
               try:
                  paths_gleam = SkyView.get_images(position=object_coord.to_string('hmsdms'),
                                  coordinates='J2000',
                                  survey='GLEAM {0}'.format(band))[0]
                  #print(links_gleam)
                  print("download %s ..." % source_names[m])
                  paths_gleam.writeto("%s/%s/%s/%s" % (args.outdir, clns[cln], surveys[args.surveys], fits_fn), overwrite=True)
               except:
                  print('Not found ', source_names[m])#name)
                  tags_non.append("{},{},{},{},{}".format(m,labels[m],source_names[m],ras[m],decs[m]))
                                              

resultsData_non = tags_non 
with open(os.path.join('./', 'NOT_FOUND_GLEAM.txt'), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))
               #os.system('wget %s -O %s/%s/%s/%s' % (links_gleam, args.outdir, clns[cln], surveys[args.surveys], fits_fn))
               #SkyView.get_images(position=object_coord,
               #                   coordinates='J2000',
               #                   survey='TGSS ADR1',
               #                   radius=radius)[0][0])
