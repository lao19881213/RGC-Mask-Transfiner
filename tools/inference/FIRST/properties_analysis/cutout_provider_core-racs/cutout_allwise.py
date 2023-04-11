import os
import re

import astropy.table as at
import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.units import Quantity
from astropy.wcs import WCS
import argparse

def WISE_cutout(position, image_size=None, filter=None):
    """
    Download WISE image cutout from IRSA
    Parameters
    ----------
    :position : :class:`~astropy.coordinates.SkyCoord`
        Target centre position of the cutout image to be downloaded.
    :image_size: int: size of cutout image in pixels
    :filter: str: Panstarrs filter (g r i z y)
    Returns
    -------
    :cutout : :class:`~astropy.io.fits.HDUList` or None
    """

    band_to_wavelength = {
        "W1": "3.4e-6",
        "W2": "4.6e-6",
        "W3": "1.2e-5",
        "W4": "2.2e-5",
    }

    url = f"https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+{position.ra.deg}+{position.dec.deg}+0.002777&RESPONSEFORMAT=CSV&BAND={band_to_wavelength[filter]}&FORMAT=image/fits"
    r = requests.get(url)
    url = None
    for t in r.text.split(","):
        if t.startswith("https"):
            url = t[:]
            break
    data = at.Table.read(r.text, format="ascii.csv")
    exptime = data["t_exptime"][0]

    if url is not None:
        fits_image = fits.open(url)

        wcs = WCS(fits_image[0].header)
        cutout = Cutout2D(fits_image[0].data, position, image_size, wcs=wcs)
        fits_image[0].data = cutout.data
        fits_image[0].header.update(cutout.wcs.to_header())
        fits_image[0].header["EXPTIME"] = exptime

    else:
        fits_image = None

    return fits_image

parser = argparse.ArgumentParser()
parser.add_argument('--resultcsv', help='hetu results file')
parser.add_argument('--outdir', help='output directory')
args = parser.parse_args()

hetu_data = pd.read_csv(args.resultcsv)

labels = hetu_data['label'].values
ras_wise = hetu_data['RAJ2000'].values
decs_wise = hetu_data['DEJ2000'].values
ras = hetu_data['centre_ra'].values
decs = hetu_data['centre_dec'].values
source_names = hetu_data['source_name'].values
boxs = hetu_data['box'].values


clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

tags_non = []

for m in range(len(labels)):
        for cln in clns.keys():
            #print(cln)
            if(labels[m]==cln):
               #print(cln)
               fits_fn = '%s_%s' % ('WISE', source_names[m])
               if os.path.isfile(f'%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'WISE', fits_fn, 'w1')):
                  print('%s_%s.fits already exists!' % ('WISE', fits_fn))
               else:
                  try:
                     position = SkyCoord(ra=ras_wise[m], dec=decs_wise[m], unit=(u.deg, u.deg), frame='fk5')
                     hud = WISE_cutout(position=position, image_size=350, filter='W1')

                     hud.writeto('%s/%s/%s/%s_%s.fits' % (args.outdir, clns[cln], 'WISE', fits_fn, 'w1'))
                  except:
                     print('Not found ', source_names[m])#name)
                     tags_non.append("{},{},{},{},{}".format(m,labels[m],source_names[m],ras[m],decs[m]))


resultsData_non = tags_non
with open(os.path.join('./', 'NOT_FOUND_WISE.txt'), 'w') as fn:
     fn.write(os.linesep.join(resultsData_non)) 
