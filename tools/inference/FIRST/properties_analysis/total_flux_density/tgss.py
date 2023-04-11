
from astropy.io import fits
import astropy.wcs as wcs
from astropy import coordinates
import astropy.units as u
import pandas as pd


def find_bbox_flux(bbox, fitsfile):
    hdu = fits.open(fitsfile)[0]

    # Set any NaN areas to zero or the interpolation will fail
    hdu.data[np.isnan(hdu.data)] = 0.0

    # Get vital stats of the fits file
    bmaj = hdu.header["BMAJ"]
    bmin = hdu.header["BMIN"]
    bpa = hdu.header["BPA"]
    xmax = hdu.header["NAXIS1"]
    ymax = hdu.header["NAXIS2"]
    try:
        pix2deg = hdu.header["CD2_2"]
    except KeyError:
        pix2deg = hdu.header["CDELT2"]
    # Montaged images use PC instead of CD
    if pix2deg == 1.0:
        pix2deg = hdu.header["PC2_2"]
    beamvolume = (1.1331 * bmaj * bmin)
    x1 = float(bbox.split('-')[0])
    y1 = float(bbox.split('-')[1])
    x2 = float(bbox.split('-')[2])
    y2 = float(bbox.split('-')[3])

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    box_data = hdu.data[hdu.data.shape[0]-y2:hdu.data.shape[0]-y1,x1:x2]
    int_flux = np.sum(box_data) #Jy/pix
    int_flux = int_flux * (pix2deg**2) / beamvolume #Jy
    return int_flux


hetu_csv = pd.read_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper.csv')

ras = hetu_csv['centre_ra'].values
decs = hetu_csv['centre_dec'].values
boxs = hetu_csv['box'].values
labels = hetu_csv['label'].values
source_names = hetu_csv['source_name'].values

clns = {'fr1': 'FRI', 'fr2': 'FRII', 'ht': 'HT', 'cj': 'CJ'}

root_dir = '/home/data0/lbq/RGC-Mask-Transfiner/tools/inference/FIRST/properties_analysis'

for m in range(len(ras)):
    FIRST_fits = '%s/%s/FIRST/%s.fits' % (root_dir, clns[labes[m]], source_names[m])
    hdu_FIRST = fits.open(FIRST_fits)[0] 
    try:
        pix_scale_hetu = hdu_FIRST.header["CD2_2"]
    except KeyError:
        pix_scale_hetu = hdu_FIRST.header["CDELT2"]
    
    TGSS_fits = '%s/%s/TGSS/%s.fits' % (root_dir, clns[labes[m]], source_names[m])
    hdu_TGSS = fits.open(TGSS_fits)[0]
    try:
        pix_scale_tgss = hdu_TGSS.header["CD2_2"]
    except KeyError:
        pix_scale_tgss = hdu_TGSS.header["CDELT2"]

    x1 = float(boxs.split('-')[0])
    y1 = float(boxs.split('-')[1])
    x2 = float(boxs.split('-')[2])
    y2 = float(boxs.split('-')[3])

    pix_scale_hetu 
