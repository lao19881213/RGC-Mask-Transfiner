import json


import os
import sys

from astropy.io import fits

import numpy as np




hdu = fits.open('J085556.090+491113.15.fits')
img_data = hdu[0].data



hdu1 = fits.open('J085556.090+491113.15.pybdsm_gaus_model.fits')
model_data = hdu1[0].data

residual_data = img_data - model_data

hdu[0].data = residual_data

hdu.writeto('residual_bdsf.fits', overwrite=True)



