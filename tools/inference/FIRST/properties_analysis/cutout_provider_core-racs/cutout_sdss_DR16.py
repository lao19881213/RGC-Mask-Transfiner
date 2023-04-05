import urllib
from astropy import units as u
from astroquery.sdss import SDSS as astroSDSS
from astropy import coordinates
import argparse
import pandas as pd
import os
from itertools import cycle
 
parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras_sdss = hetu_data['RA_ICRS'].values
decs_sdss = hetu_data['DE_ICRS'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
source_names = hetu_data['source_name'].values
boxs = hetu_data['box'].values
objIDs = hetu_data['objID'].values


#ra = 256.073204
#dec = 60.654467 

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

tags_non = []

for m in range(len(labels)):
        for cln in clns.keys():
            #print(cln)
            if(labels[m]==cln):
               #print(cln)
               fits_fn = '%s_%s' % ('SDSS', source_names[m])
               if os.path.isfile(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, 'r')):
                  print('%s_%s.fits already exists!' % ('SDSS', fits_fn))
               else:
                  try:
                     position = coordinates.SkyCoord(ra=ras[m], dec=decs[m], unit=(u.deg, u.deg), frame='icrs') #frame='fk5')#, frame='icrs')
                      #try:               
                     result = astroSDSS.query_region(position, radius=5*u.arcsec)
                     #print(result)
                     result_hetu = result[result['objid']==objIDs[m]] 
                     print('fetching images....\n')          
                     #band='ugriz' 
                     imgs = astroSDSS.get_images(matches=result_hetu, band='r', timeout=10, data_release=16)#, coordinates='J2000')
                     #hdul_lists = astroSDSS.get_images(coordinates=position, radius = 5.0*u.arcmin, band='ugriz',  data_release=12)
                     #print(len(hdul_lists))
                     if imgs:
                        print('writing %s' % source_names[m], end='')
                        #counter = 0
                        fits_fn = '%s_%s' % ('SDSS', source_names[m])
                        for HDU, band in zip(imgs, cycle('r')):
                            print('.', end='')
                            #if os.path.isfile(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, band)):
                            HDU.writeto(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, band),
                                            overwrite=True)
                            #    if band == 'z':
                            #        counter += 1
                            #else:
                            #    HDU.writeto(f'%s/%s/%s/%s_%s_%d.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, band, counter),
                            #                overwrite=True)
                        print('\n')
                     #fits_fn = '%s_%s.fits' % ('SDSS', source_names[m])
                     #print("%s/%s/%s/%s" % (args.outdir, clns[cln], 'SDSS', fits_fn)) 
                     #hdul_lists.writeto("%s/%s/%s/%s" % (args.outdir, clns[cln], 'SDSS', fits_fn), overwrite=True)
                     #hdul_lists = astroSDSS.get_images(coordinates=position, radius = 5.0*u.arcmin, band='r',  data_release=16)
                  except:
                  #else:
                     try:
                        position = coordinates.SkyCoord(ra=ras_sdss[m], dec=decs_sdss[m], unit=(u.deg, u.deg), frame='icrs')
                         #try:               
                        result = astroSDSS.query_region(position, radius=5*u.arcsec)
                        #print(result)
                        result_hetu = result[result['objid']==objIDs[m]]
                        print('fetching images....\n')
                        #band='ugriz' 
                        imgs = astroSDSS.get_images(matches=result_hetu, band='r', timeout=10, data_release=16)#, coordinates='J2000')
                        #hdul_lists = astroSDSS.get_images(coordinates=position, radius = 5.0*u.arcmin, band='ugriz',  data_release=12)
                        #print(len(hdul_lists))
                        if imgs:
                           print('writing %s' % source_names[m], end='')
                           #counter = 0
                           fits_fn = '%s_%s' % ('SDSS', source_names[m])
                           for HDU, band in zip(imgs, cycle('r')):
                               print('.', end='')
                               #if os.path.isfile(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, band)):
                               HDU.writeto(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, band),
                                               overwrite=True)
                               #    if band == 'z':
                               #        counter += 1
                               #else:
                               #    HDU.writeto(f'%s/%s/%s/%s_%s_%d.fits' % (args.outdir, clns[cln], 'SDSS', fits_fn, band, counter),
                               #                overwrite=True)
                           print('\n')
                     except:
                        print('Not found ', source_names[m])#name)
                        tags_non.append("{},{},{},{},{}".format(m,labels[m],source_names[m],ras[m],decs[m]))


resultsData_non = tags_non
with open(os.path.join('./', 'NOT_FOUND_SDSS.txt'), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non))
