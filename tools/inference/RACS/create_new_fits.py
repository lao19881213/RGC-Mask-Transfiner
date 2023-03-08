import os,sys
import numpy as np
import pdb
from astropy.io import fits
import astropy.wcs as wcs
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inpdir', dest='inpdir', type=str, default='./', help='pred input png file directory')
parser.add_argument('--outdir', dest='outdir', type=str, default='./', help='output png file directory')
args = parser.parse_args()


outdir = args.outdir
input_dir = args.inpdir
file_nms = os.listdir(input_dir)
for fn in file_nms:
    if not fn.endswith('.fits'):
       continue
   
    fits_file = fn
    hdu0 = fits.open(os.path.join(input_dir, fits_file))
    hdu = fits.open(os.path.join(input_dir, fits_file))[0]
    print(hdu.data.shape)
    img_data = hdu.data
    head0 = hdu0[0].header
    head0.remove('NAXIS3')
    head0.remove('NAXIS4')
    head0.remove('CTYPE3')
    head0.remove('CDELT3')
    head0.remove('CRVAL3')
    head0.remove('CRPIX3')
    head0.remove('CUNIT3')
    head0.remove('CTYPE4')
    head0.remove('CDELT4')
    head0.remove('CRVAL4')
    head0.remove('CRPIX4')
    #head0.remove('CUNIT4')
    #head0.remove('PC01_03')
    #head0.remove('PC01_04')
    #head0.remove('PC02_03')
    #head0.remove('PC02_04')
    #head0.remove('PC03_01')
    #head0.remove('PC03_02')
    #head0.remove('PC03_03')
    #head0.remove('PC03_04')
    #head0.remove('PC04_01')
    #head0.remove('PC04_02')
    #head0.remove('PC04_03')
    #head0.remove('PC04_04')
    hdu0[0].data = img_data.reshape(hdu.data.shape[0],hdu.data.shape[1])
    head0.set('NAXIS',2)
    hdu0.writeto(os.path.join(outdir, "%s.fits" % os.path.splitext(fits_file)[0]), overwrite=True)
    print("Successful generate %s" % fn)

