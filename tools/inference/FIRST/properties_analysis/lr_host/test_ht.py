#
import numpy as np, os, argparse
from astropy.table import Table, join, unique, MaskedColumn, setdiff
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.utils.tap.core import TapPlus
from astroquery.xmatch import XMatch
from astropy.io import ascii
from distutils.util import strtobool
import time
import pandas as pd
from lr_host_finding.find_hosts_lr import *

wise_psf = 1.8*u.arcsec
double_poscal = 6 ##calibration errors in arcsec

def find_allwise(data, racol='RA', decol='DEC', maxsep=30*u.arcsec,
                 catcols=['AllWISE', 'W1mag', 'e_W1mag', 'W2mag',
                          'e_W2mag', 'W3mag', 'e_W3mag', 'W4mag',
                          'e_W4mag'],
                 namecol='Name', cat2='vizier:II/328/allwise',
                 timeout=300, sepcol_name='Sep_AllWISE', chunksize=50000):
    'use cds xmatch to query sdss'
    ###try to replace with async query (might not need to)
    xm = XMatch()
    xm.TIMEOUT = timeout
    
    if len(data)>chunksize:
        print('')
        print('Upload data is large, querying AllWISE in chunks')
        job_results = []
        n_chunks = int(np.ceil(len(data)/chunksize))
        for i in range(n_chunks):
            print('')
            print('AllWISE query chunk ' + str(i+1) + '/' + str(n_chunks))
            print('')
            data_chunk = data[i*chunksize: (i+1)*chunksize]
            job = xm.query_async(cat1=data_chunk, cat2=cat2, max_distance=maxsep,
                                 colRA1=racol, colDec1=decol)
            job_results.append(ascii.read(job.content.decode()))
        xmatch = vstack(job_results)
    else:
        xmatch = xm.query_async(cat1=data, cat2=cat2, max_distance=maxsep,
                                colRA1=racol, colDec1=decol)
        xmatch = ascii.read(xmatch.content.decode())
    #print(xmatch)                  
    xmatch.rename_column(name='angDist', new_name=sepcol_name)
    ###add in sep col units
    xmatch[sepcol_name] = np.round(xmatch[sepcol_name], 2)
    xmatch[sepcol_name].unit = maxsep.unit
    
    if len(xmatch)>0:
        ###add in missing (no match within search radius) here!
        found = np.unique(xmatch[namecol])
        missed = [i for i in data[namecol] if i not in found]
        missed = Table({namecol: missed})
        if len(missed) > 0:
           missed = join(missed, data, keys=namecol, join_type='inner')
           xmatch = vstack([xmatch, missed])
           xmatch.sort(sepcol_name)
        else:
           xmatch.sort(sepcol_name)
        best_match = unique(xmatch, namecol) ###ensures no duplicates
        bestcols = data.colnames + catcols + [sepcol_name]
        best_match = best_match[bestcols]
        best_match.sort(racol)
        for col in best_match.colnames:
            if col in data.colnames:
                best_match[col].unit = data[col].unit
        
        xmatch.sort(racol)
        rcols = data.colnames
        rcols.remove(namecol)
        xmatch.remove_columns(names=rcols)
    
        return best_match, xmatch
    
    else:
        rcols = data.colnames
        rcols.remove(namecol)
        xmatch = xmatch.remove_columns(names=rcols)
        return xmatch


#ht_csv = pd.read_csv('ht.csv')
#print(ht_csv)

#pix_resolution = 1.8 # arcsec
#
#x1 = float(boxs[m].split('-')[0])
#y1 = float(boxs[m].split('-')[1])
#x2 = float(boxs[m].split('-')[2])
#y2 = float(boxs[m].split('-')[3])
#
#
#x1 = int(x1)
#y1 = int(y1)
#x2 = int(x2)
#y2 = int(y2)
#data = 'ht.csv'
data = Table.read('ht.csv', format='csv')
#data = table
print(data)
host_data = find_allwise(data, racol='RA', decol='DEC', maxsep=30*u.arcsec,
             catcols=['AllWISE', 'W1mag', 'e_W1mag', 'W2mag',
             'e_W2mag', 'W3mag', 'e_W3mag', 'W4mag', 'e_W4mag'],
             namecol='Name', cat2='vizier:II/328/allwise',
             timeout=300, sepcol_name='Sep_AllWISE', chunksize=50000)


hostdata = {}

hostdata['sources'] = host_data[0]
hostdata['hosts'] = host_data[1]

print(type(hostdata['hosts']['AllWISE']))

if type(hostdata['hosts']['AllWISE'])!=MaskedColumn:
   hostdata['hosts']['AllWISE'] = MaskedColumn(hostdata['hosts']['AllWISE'])

#hostdata['sources'] = host_data[0]
#hostdata['hosts'] = host_data[1]

print(type(hostdata['hosts']['AllWISE']))

print(hostdata['hosts'])

image_info = Table.read('lr_host_finding/AllWISE-image_atlas_metadata-lite.fits')

if type(hostdata['hosts']['AllWISE'])==MaskedColumn:
   hostdata['hosts'] = hostdata['hosts'][~hostdata['hosts']['AllWISE'].mask]

outdir = 'output_files'

print(host_data[0]['RA'])
host_matching(source_cat=host_data[0], host_cat=host_data[1],
             image_meta=image_info,
             sra='RA', sdec='DEC', sid='Name', sflux='Flux',
             ssize='LAS',
             iid='coadd_id', ira='ra', idec='dec',
             hid='AllWISE', hra='RAJ2000', hdec='DEJ2000',
             hmags=['W1mag'], hmagerrs=['e_W1mag'],
             sepcol='Sep_AllWISE',
             bin_dir='lr_bin', outdir=outdir,
             assumed_psf=wise_psf, verbose=True,
             radio_beam_size=6, cal_errors=double_poscal) #radio beam size, VLASS: 3'', FIRST 6''
