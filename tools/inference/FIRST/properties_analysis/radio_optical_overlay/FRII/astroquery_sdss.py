# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:08:02 2023

@author: Baoqiang Lao
"""
# Required to see plots when running on mybinder
import matplotlib 
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline 

# Python standard-libraries to download data from the web
from urllib.parse import urlencode
from urllib.request import urlretrieve

#Some astropy submodules that you know already
from astropy import units as u
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.io import fits


#only here to display images
from IPython.display import Image

# These are the new modules for this notebook
from astroquery.simbad import Simbad
from astroquery.sdss import SDSS

from astroquery.sdss import SDSS
from astropy import coordinates as coords
#pos = coords.SkyCoord('0h8m05.63s +14d50m23.3s', frame='icrs')
#pos = coords.SkyCoord('00h12m47.57s +00d47m15.8s', frame='icrs')
#pos = coords.SkyCoord('00h21m07.62s -00d55m31.4s', frame='icrs')
pos =  coords.SkyCoord(ra=205.00924, dec=0.575329033996947, unit=(u.deg, u.deg), frame='fk5')
xid = SDSS.query_region(pos, radius='5 arcsec', spectro=True, data_release=17)
sp = SDSS.get_spectra(matches=xid, data_release=17)
#im = SDSS.get_images(matches=xid, band='r')
# im[0].writeto('test.fits')
"The spectrum is stored as a table in the second item of the list."
"That means that we can get the Table doing the following"
spectra_data = sp[0][1].data

flux = spectra_data['flux']

plt.figure(1)
plt.plot(10**spectra_data['loglam'], spectra_data['flux'])
plt.xlabel('wavelenght (Angstrom)')
plt.ylabel('flux (nanomaggies)')
plt.title('SDSS spectra of xxx')

plt.show()

# The fourth record stores the positions of some emission lines

lines = sp[0][3].data

for n in ['[O_II] 3727', '[O_III] 5007', 'H_alpha']:
    print(n, " ->", lines['LINEWAVE'][lines['LINENAME']==n])

plt.figure(2)    
plt.plot(10**spectra_data['loglam'], spectra_data['flux'], color='black')
plt.axvline(x=lines['LINEWAVE'][lines['LINENAME']=='[O_II] 3727'], label=r'O[II]', color='blue')
plt.axvline(x=lines['LINEWAVE'][lines['LINENAME']=='[O_III] 5007'], label=r'O[III]', color='red')
plt.axvline(x=lines['LINEWAVE'][lines['LINENAME']=='H_alpha'], label=r'H$\alpha$', color='green')

plt.xlabel('wavelenght (Angstrom)')
plt.ylabel('flux (nanomaggies)')
plt.title('SDSS spectra of xxx')
plt.legend()  

#Calculate F[o_III]  
# from astroquery.sdss import SDSS
##Method 1
# query = "SELECT\ em.Flux_OIII_5006 \
#             FROM SpecObjAll AS spec \
#             JOIN SpecPhotoAll AS spho ON spec.specObjID = spho.specObjID \
#             JOIN emissionLinesPort AS em ON em.specObjID = spec.specObjID \
#             WHERE spec.ra = 2.02629 AND spec.dec = 0.011883"

# results = SDSS.query_sql(query).to_pandas()

#Method 2 
float_lam = 10**spectra_data['loglam'] 
int_lam =  float_lam.astype(np.int)
x=lines['LINEWAVE'][lines['LINENAME']=='[O_III] 5007']
index_oIII = np.where(int_lam==int(x))[0]
flux_oIII = flux[index_oIII][0]
print('flux of oIII -> ',flux_oIII)

#plt.show()

#redshift
redshift = sp[0][2].data['Z'][0]
print('redshift -> ', redshift)

# RA and DEC
ra=sp[0][0].header['RA'] ; dec=sp[0][0].header['DEC']
print('RA -> ', ra)
print('Dec -> ', dec)

#velocity dispersion in km/s
vdisp = sp[0][2].data['VDISP'][0]
print('Stellar velocity dispersion -> ', vdisp)

# black hole of mass
M_BH = 8.13 + 4.02*np.log10(vdisp/200.0)
print('black hole of mass -> ', M_BH)

#Concentration index
#xid['specobjid']
query = "SELECT\
        TOP 1 p.petroR50_r, p.petroR90_r \
        FROM PhotoObjAll AS p JOIN specObjAll s ON s.bestobjid = p.objid \
        WHERE s.specobjid = %s" % (xid['specobjid'][0]) 
SDSSpetro = SDSS.query_sql(query, data_release=17)
Ci =SDSSpetro['petroR90_r'][0]/SDSSpetro['petroR50_r'][0]
print('Concentration index -> ', Ci)
