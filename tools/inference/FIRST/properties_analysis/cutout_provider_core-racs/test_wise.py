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
from astroquery.hips2fits import hips2fits
from astroquery.mast import Observations
from astroquery.skyview import SkyView
from pyvo.dal import sia

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


position = SkyCoord(ra=125.1072966, dec=46.5902029, unit=(u.deg, u.deg), frame='fk5')
hud = WISE_cutout(position=position, image_size=350, filter='W1')

hud.writeto('test_wise.fits')
