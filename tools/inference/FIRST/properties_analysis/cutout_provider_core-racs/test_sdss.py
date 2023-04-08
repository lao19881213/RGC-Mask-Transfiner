import urllib
from astropy import units as u
from astroquery.sdss import SDSS as astroSDSS
from astropy import coordinates
import argparse
import pandas as pd
import os

position = coordinates.SkyCoord(ra=15.089488659906026,dec = 15.089488659906026, unit=(u.deg, u.deg), frame='fk5')



result = astroSDSS.query_region(position, radius=5*u.arcsec) #, data_release=16)

imgs = astroSDSS.get_images(matches=result, band='r', timeout=10, data_release=16, coordinates='J2000')

id = result[result['objid']==1237668554291348043]
