import cv2
import pycocotools.mask as cocomask
from skimage.measure import find_contours
import pandas as pd
import os 
import numpy as np
from matplotlib.patches import Polygon
from scipy import spatial
from astropy.io import fits
import astropy.wcs as wcs

FIRST_result = '/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_vlass.csv'
fits_dir = '/home/data0/lbq/inference_data/FIRST_fits'

hetu_csv = pd.read_csv(FIRST_result)
pngs = hetu_csv['image_filename'].values
masks = hetu_csv['mask'].values
LAS = []
#angular size
for m in range(len(pngs)):
    hdu = fits.open(os.path.join(fits_dir, os.path.splitext(pngs[m])[0]+'.fits'))[0]
    w1=wcs.WCS(hdu.header, naxis=2) 
    image = cv2.imread(os.path.join(fits_dir, pngs[m]))
    (height, width) = image.shape[:2]
    segm = {
            "size": [width, height],
            "counts": masks[m]}
    mask = cocomask.decode(segm)
    padded_mask = np.zeros(
         (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    ps = []
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts)
        ps.extend(p.xy)
    ps = np.array(ps)
    pts = ps
    candidates = pts[spatial.ConvexHull(pts).vertices]
    # get distances between each pair of candidate points
    dist_mat = spatial.distance_matrix(candidates, candidates)
    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    #print(candidates[i], candidates[j])
    x_r = abs(candidates[i][0]-candidates[j][0])
    y_r = abs(candidates[i][1]-candidates[j][1])
    #print(np.sqrt(x_r**2+y_r**2))
    RA1_degree, DEC1_degree = w1.wcs_pix2world([[candidates[i][0], candidates[i][1]]], 0)[0]
    RA2_degree, DEC2_degree = w1.wcs_pix2world([[candidates[j][0], candidates[j][1]]], 0)[0]
    #theta = np.arccos(np.sin(dec1)*np.sin(dec2)+np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)) 
    angularDist_arcsec=np.degrees(np.arccos(np.sin(np.radians(DEC1_degree),dtype=np.float64)*np.sin(np.radians(DEC2_degree),dtype=np.float64) + np.cos(np.radians(DEC1_degree),dtype=np.float64)*np.cos(np.radians(DEC2_degree),dtype=np.float64)*np.cos(np.radians(RA1_degree-RA2_degree),dtype=np.float64)))*3.6E3
    print(angularDist_arcsec, angularDist_arcsec/60.0)
    LAS.append(angularDist_arcsec/60.0)


hetu_csv['LAS'] = LAS 
hetu_csv.to_csv('/home/data0/lbq/RGC-Mask-Transfiner/FIRST_results/FIRST_HeTu_paper_fr2_flux_fixed_vlass.csv', index = False)
